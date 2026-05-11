# Abstract-CoT (arXiv:2604.22709v2) — Smoke Repro on Qwen3-4B

**Date:** 2026-05-10 → 2026-05-11
**Scope:** Reproduce Baseline vs Abstract-CoT (Warm-up) on MATH-500 for Qwen3-4B.
**Status:** Pipeline verified end-to-end. Smoke trained on 5k Dolci-Think samples (T=1 PI round, 1 epoch/phase). No RL.

---

## Headline numbers

| Method | MATH-500 acc | Mean tokens (reasoning + response) |
|---|---|---|
| Paper Baseline (Qwen3-4B, verbal CoT) | 83.2 | 1087 |
| **Our Baseline** (same setup) | **83.60** | **1067** |
| Paper Abstract-CoT (Warm-up) | 86.2 | 168 |
| **Our smoke Warm-up** (T=1, 5k, 1ep, LoRA, temp=0.7, m_min=16) | **73.20** | **433** |

- **Baseline reproduces paper** (within noise of 0.4 pts / 20 tokens) — calibration is solid.
- **Smoke Warm-up is ~10 pts below the paper's Warm-up** and ~2.7× more tokens.
- The gap is consistent with the shortcuts we took (see "Workarounds" below).

---

## Hardware actually available

The environment has 4× A100-SXM4-40GB. ECC and load-test status (re-checked 2026-05-11 after the smoke run):

```
GPU 0: A100-SXM4-40GB  vol ECC unc: 0  agg ECC unc: 933   matmul OK, vLLM OK
GPU 1: A100-SXM4-40GB  vol ECC unc: 0  agg ECC unc: 20    matmul OK, vLLM OK
GPU 2: A100-SXM4-40GB  vol ECC unc: 8  agg ECC unc: 3640  matmul OK on re-test; faulted vLLM earlier
GPU 3: A100-SXM4-40GB  vol ECC unc: 2  agg ECC unc: 20    matmul still fails (cudaErrorECCUncorrectable)
```

**Three GPUs are usable: 0, 1, 2.** GPU 3 reliably fails. GPU 2 is recovered and passes both a sustained bf16 matmul stress test (200 iters of 8192×8192) and a 3-GPU matmul check; its high aggregate ECC count is from previous runtime errors and isn't recurring right now. The original smoke run only used GPUs 0–1 because of an early vLLM crash on GPU 2 that didn't reproduce on re-test.

**Note on vLLM tensor parallelism:** Qwen3-4B has 32 attention heads, so vLLM `tensor_parallel_size` must divide 32 → only TP ∈ {1, 2, 4, 8, 16} are valid. With 3 healthy GPUs we can run **eval at TP=2** (leaving 1 GPU idle) or **TP=1 with replicated serving across 3 instances**. Training (data-parallel) has no such constraint and benefits linearly from 3 GPUs.

**Effective compute: 3× A100-40GB for training, TP=2 for vLLM eval.** Paper used **8× H100-80GB for SFT, up to 32× H100 for RL.** Roughly 1/8th the SFT FLOPs and 1/32nd the RL FLOPs.

---

## Workarounds taken (everything that diverges from the paper)

Listed in rough order of impact on the final number.

### 1. **LoRA fallback instead of full fine-tuning** (biggest divergence)

Tried full FT first. Two attempts failed:
- Single-GPU full FT: instant OOM (Adam fp32 states alone need ~32 GB on top of 8 GB params, exceeding 40 GB).
- DeepSpeed ZeRO-3 across 2 GPUs (no offload): trained at ~16 s/step with constant `pytorch_allocator_cache_flush` warnings (peak 36.7 GB / 40 GB). At that rate, **a single 5k-1-epoch phase projects to ~83 min, vs. ~33 min with LoRA.** Worse, sequence length had to be cut to 1024 to fit, which truncates the median 18.8k-token Dolci CoT down to a stub.
- DeepSpeed ZeRO-3 with `offload_optimizer: cpu` would have fixed the memory issue but **DeepSpeed's CPU Adam extension fails to build in this environment** (`AttributeError: DeepSpeedCPUAdam has no attribute ds_opt_adam`, traceable to nvcc/CUDA-version mismatch with torch 2.11 + cu13).

Final setup used:
- `peft` LoRA, `r=32`, `alpha=64`, target modules `{q,k,v,o,gate,up,down}_proj` (7 projections × 36 layers)
- `modules_to_save=["embed_tokens", "lm_head"]` so the new abstract-vocab rows can move freely. Note Qwen3-4B has `tie_word_embeddings=True`; merging untied them and set `tie_word_embeddings=False` in the saved config. peft emitted a `ensure_weight_tying` warning that's worth revisiting before scaling up.
- Trainable params: **842.9 M / 4.86 B (17.3%)** — mostly the embedding rows (388 M each for input + output after untying).
- Optimizer: torch `AdamW` (fused) directly, no DeepSpeed, no offload.
- Parallelism: vanilla 2-GPU DDP via `accelerate launch --num_processes 2 --mixed_precision bf16`.
- Saved as a peft adapter, then `merge_lora.py` merges into a full bf16 model for the next stage.

**Why this matters for the result:** paper does full fine-tuning, where the model's existing math knowledge can be reshaped to actually use abstract tokens as a reasoning substrate. With LoRA + ~17% trainable params, the model retains too much of its base "answer-from-prompt" reflex and the abstract trace doesn't have to carry signal.

### 2. **5k samples instead of 600k**

Paper uses a 600k subsample of `allenai/Dolci-Think-SFT-7B`. We used 5k for the smoke. **120× less data.**

Sub-issue: streaming the dataset and taking the first N gives a homogeneous slice (all Tulu-Wildchat in the first parquet shards, almost all with empty `<think>` blocks). Used `ds.shuffle(seed=42, buffer_size=20000)` + a filter for `len(cot) >= 200` and got a 64% math (OpenThoughts3) / 36% python mix.

### 3. **T=1 PI round instead of T=3**

Paper iterates 3 times (Algorithm 1):
- **t=1:** random abstract traces → bottleneck SFT → self-distill (the abstract trace is meaningless on entry, so this phase mostly teaches "predict Y from X, ignoring Z").
- **t≥2:** abstract traces are now generated *on-policy via constrained decoding from the just-trained model*, so they start to carry signal. Bottleneck SFT under that pressure is where Z̃ becomes a real bottleneck.

Our smoke only does t=1, so the model never sees a Z̃ that's been shaped by the bottleneck. This is the single biggest reason the abstract trace doesn't replace verbal CoT.

### 4. **1 epoch per phase instead of 3**

Paper does 3 epochs of bottleneck SFT and 3 epochs of self-distill per round. We did 1 each. **3× less training per phase.**

### 5. **No RL stage at all**

Paper warm-starts GRPO with `gpt-oss-20b` as a generative reward model. We stopped at warm-up because the user asked for SFT-only comparison. The paper's "Abstract-CoT (Warm-up)" row in Table 1 is also pre-RL, so this is *not* an apples-to-oranges concern for the comparison itself — but it does mean we lose the RL exploration that pushes the model away from "emit empty/trivial Z̃."

### 6. **Sequence-length truncation**

Dolci-Think CoTs are *long*: median 18,811 tokens, p90 28,214. Our `max_len`:
- Full FT attempt: 1024 → only 112/5000 examples fit without CoT truncation.
- LoRA final run: 2048 → only 359/5000 fit.
- The other ~93% have CoT truncated from the right.

When CoT is heavily truncated, the bottleneck is essentially "fit a tiny fragment of reasoning into ~64 abstract tokens" — weaker signal than the paper assumes. Paper does not state `max_len` explicitly, but with H100-80GB and ZeRO-3 they likely run at 8k–16k.

### 7. **Greedy abstract decoding initially missed special tokens**

vLLM strips tokens added with `add_tokens(..., special_tokens=True)` from `.text` by default. First abstract eval reported `mean_abs=0.0` and I mis-diagnosed this as an "empty-trace failure." It was a counting bug — corrected by passing `skip_special_tokens=False` to `SamplingParams` and counting via `outputs[0].token_ids` instead of re-tokenizing the decoded text.

After correction: greedy abstract trace = 15.6 tokens, sampled (temp 0.7) = 22 tokens with `m_min=16`, 38 with `m_min=32`. Accuracy moves by ≤1.2 pts across these settings.

### 8. **gen_traces uses HF `generate`, not vLLM**

The on-policy abstract trace generation between Phase A and Phase B needs a stateful constrained-decode logits processor (force V_abs tokens until `<endabstract>` or `m_max`, then stop). Implemented as a `LogitsProcessor` in `code/abstract.py` and run via HF `model.generate()`. Single GPU, batch 16, ~7.6 samples/s → 11 min for 5k.

vLLM could probably be coerced into the same shape but the integration cost wasn't worth it for the smoke. **For the full run this is a real bottleneck** — at this rate gen_traces on 60k examples is ~2.2 hr per PI round, all serial single-GPU. Switching to vLLM here is the single best engineering improvement available.

### 9. **MATH-only evaluation**

Per user instruction. Paper evals MATH-500, AlpacaEval-LC-2.0, HotpotQA. We only ran MATH-500. AlpacaEval-LC needs an LLM judge (GPT-4-class) we don't have set up; HotpotQA is straightforward (F1) but was out of scope.

### 10. **Minor**: embedding initialization

Paper doesn't specify how the new V_abs rows are initialized. We used **mean of existing embeddings + N(0, 1e-3) noise** for both `embed_tokens` and `lm_head` rows. Alternative initializations (zeros, random scaled to existing norm) are plausible.

### 11. **Minor**: short Phase A LR schedule

`get_cosine_schedule_with_warmup` was set with `total_steps` computed *before* `accelerator.prepare()`, so the per-process step count was wrong and the cosine wrapped around. LR went 1e-4 → ~5e-7 → back up to ~1e-4. Functionally a triangular schedule. Hasn't been observed to hurt the smoke loss curve but should be fixed before scaling up.

---

## Smoke timing (5k examples, 1 epoch / phase, LoRA, 2× A100-40GB)

| Stage | Wall | Notes |
|---|---|---|
| Phase A (bottleneck SFT, random Z̃) | **33 min** | 156 opt steps × 12.7 s, seq_len 2048 |
| Merge LoRA → full model | ~30 s | |
| `gen_traces` (5k, no-CoT, m_max=128) | **11 min** | HF generate, 1 GPU, batch 16, ~7.6/s |
| Phase B (self-distill, on-policy Z̃) | **15.5 min** | 156 opt steps × 6 s, shorter seq (no CoT) |
| Merge LoRA → full model | ~30 s | |
| Eval MATH-500 (abstract mode, vLLM) | ~1 min | 500 problems, tp=2 |
| **Total smoke run (excluding setup)** | **~62 min** | |

Plus initial setup that won't repeat: pip installs (~10 min), Qwen3-4B + MATH-500 + 5k Dolci download (~5 min), `extend_model.py` (~30 s), baseline eval calibration (~3 min) → total project session: ~85 min.

---

## Estimated time for "full SFT"

Three interpretations of "full SFT," with timings extrapolated from the measured smoke step rates. **Original estimates (italicised) assumed 2 GPUs; updated estimates assume 3 GPUs for training (gen_traces and eval also benefit but less).**

Scaling assumptions:
- Training step time on 3 GPUs ≈ (2/3) × 2-GPU step time (~linear DDP scaling for LoRA at this size; slightly worse for full FT due to higher comm).
- `gen_traces` can be sharded 3-way across GPUs trivially → ~3× speedup possible (current code is single-GPU, needs a small change to split data).
- vLLM eval pinned to TP=2 regardless (head-count constraint).

### Interpretation A — Full **fine-tuning** (no LoRA), same 5k data, T=1, 1 epoch

Just removing the LoRA workaround, everything else equal.

- DeepSpeed ZeRO-3 (no offload) measured **~16 s/step at seq_len 1024** on this 2-GPU box, with significant cache pressure.
- 5k / 16 grad_accum / 2 GPUs = 156 opt steps → 156 × 16 = **~42 min for Phase A**.
- Phase B (no CoT in sequence, faster): **~20 min**.
- gen_traces unchanged: 11 min.
- **Full-FT smoke total: ~75 min** (vs. our 62 min with LoRA).
- Requires `max_len ≤ 1024` to fit in 40 GB. CoT will be truncated for ~98% of examples.

Caveat: this would also require fixing DeepSpeed CPU Adam, or accepting that ~93% of CoTs get truncated. As-is, this run is technically possible but the bottleneck signal is severely degraded.

### Interpretation B — LoRA, but at "real" data/PI/epoch scale

Keep LoRA. Scale up data to 60k, T=3 rounds, 3 epochs per phase. The most realistic mid-fidelity run on this hardware.

| Stage | Per-PI-round cost | × T=3 |
|---|---|---|
| Phase A (60k × 3 ep) | 33 min × (60k/5k) × 3 ≈ ~20 hr | ~60 hr |
| gen_traces (60k) | 11 min × 12 ≈ ~2.2 hr | ~4.4 hr (only 2 rounds need on-policy) |
| Phase B (60k × 3 ep) | 15.5 min × 12 × 3 ≈ ~9.3 hr | ~28 hr |
| Merges | ~1 min × 6 | negligible |

**Estimated total: ~92 hr ≈ ~4 days** of continuous training.

Caveat: gen_traces is serial single-GPU HF generate; would benefit substantially from a vLLM port (probably 4–6× speedup).

### Interpretation C — Full FT, paper-scale (600k, T=3, 3 epochs)

This is what "matches the paper" looks like:

- Per Phase A epoch: 600k / (16 grad_accum) / 2 GPUs = 18,750 opt steps × 16 s = **~83 hr**
- 3 epochs × 3 rounds = 9 Phase A epochs ≈ **~750 hr**
- Phase B (faster): 9 × ~37 hr ≈ **~335 hr**
- gen_traces × 2 on-policy rounds × 600k: 11 min × 120 × 2 ≈ **~44 hr**

**Estimated total: ~1130 hr ≈ ~47 days** of continuous training on 2× A100-40GB. Not feasible here.

### Summary table

Estimates updated for 3 healthy GPUs (training DDP gains ~1.5×; gen_traces gains ~3× if parallelised; eval unchanged).

| Run | Data | T | Epochs | Adapter | 2-GPU est. | **3-GPU est.** |
|---|---|---|---|---|---|---|
| Smoke (done) | 5k | 1 | 1 | LoRA | ~60 min | ~40 min |
| Full-FT smoke | 5k | 1 | 1 | full FT | ~75 min | ~50 min |
| Mid-fidelity LoRA | 60k | 3 | 3 | LoRA | ~92 hr (~4 d) | **~60 hr (~2.5 d)** |
| Mid-fidelity full FT | 60k | 3 | 3 | full FT | ~120 hr (~5 d) | **~80 hr (~3.3 d)** (needs CPU-Adam fix) |
| Paper-scale full FT | 600k | 3 | 3 | full FT | ~1130 hr (~47 d) | ~770 hr (~32 d) — still not feasible |

**Recommendation:** mid-fidelity LoRA (60k × T=3 × 3 epochs) on 3 GPUs is the right next run — **~2.5 days wall.** Biggest expected delta is going from T=1 → T=3 where the on-policy abstract traces start carrying signal. Before launching, four things to fix:
1. Switch training launch to `--num_processes 3` and shard `gen_traces` 3-way (or port it to vLLM TP=1 across 3 replicas, ~5–10× speedup).
2. The cosine LR schedule (`total_steps` math).
3. Tighten the `tie_word_embeddings` handling in peft to avoid silently untying at merge.
4. Confirm GPU 2 stays stable over a multi-hour run (rerun the bf16 stress loop for ~10 min before kicking off the long job).

---

## File layout (under `/workspace/`)

```
code/
  abstract.py            # V_abs definitions, LogitsProcessor
  data_utils.py          # bottleneck/distill seq construction, 4-D block mask
  extend_model.py        # tokenizer + embedding extension
  train_phase_lora.py    # one SFT phase (bottleneck or distill) under DDP+LoRA
  train_phase.py         # same, but under DeepSpeed ZeRO-3 (full FT path, currently superseded)
  gen_traces.py          # on-policy abstract trace generation (HF generate)
  merge_lora.py          # peft adapter + base → merged full model
  eval_math.py           # vLLM MATH-500 eval, baseline and abstract modes
  ds_zero3.json          # DeepSpeed config (unused after LoRA switch)
  accelerate_ds.yaml     # accelerate config (DeepSpeed path)
  env.sh                 # HF_HOME, CUDA_VISIBLE_DEVICES=0,1
data/
  math500.jsonl          # 500 problems
  dolci_5k.jsonl         # 5k filtered Dolci-Think-SFT examples
runs/
  baseline_math500.jsonl                  # 500 baseline rows (83.6%)
  qwen3-4b-abs/base/                      # extended-vocab Qwen3-4B
  qwen3-4b-abs/pi1_phaseA/                # LoRA adapter after Phase A
  qwen3-4b-abs/pi1_phaseA_merged/         # merged full model
  qwen3-4b-abs/traces_distill.jsonl       # 5k on-policy traces (post-Phase-A)
  qwen3-4b-abs/pi1_phaseB/                # LoRA adapter after Phase B
  qwen3-4b-abs/pi1_phaseB_merged/         # final merged Warm-up model
  abstract_math500*.jsonl                 # eval outputs (various decoding settings)
```
