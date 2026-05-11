# ThinkingWithoutWordsRepro

A clean-room reproduction of **"Thinking Without Words: Efficient Latent Reasoning with Abstract Chain-of-Thought"** (Ramji, Naseem, Fernandez Astudillo; arXiv:2604.22709v2; IBM Research AI) for **Qwen3-4B on MATH-500**.

> **Scope.** This repo implements the SFT half of the paper — the policy-iteration warm-up (Phase A: bottlenecked SFT + Phase B: self-distillation). The RL stage (GRPO with `gpt-oss-20b` as judge) is intentionally out of scope; the paper's "Abstract-CoT (Warm-up)" row in Table 1 is the comparison target. Eval is MATH-500 only.

> **Status.** Pipeline verified end-to-end (T=1, 5k samples, 1 epoch/phase, LoRA): baseline reproduces paper to within 0.4 pts; smoke warm-up runs but underperforms the paper's warm-up by ~10 pts due to known shortcuts. See [`docs/SMOKE_REPORT.md`](docs/SMOKE_REPORT.md) for a full account of workarounds, failure modes, and time estimates for scaling up.

---

## What this repo contains

- `src/` — Python modules implementing the paper's mechanism
- `scripts/` — bash drivers, one per pipeline stage + a one-shot smoke runner
- `configs/` — DeepSpeed + accelerate configs (only needed for the full-FT path)
- `docs/SMOKE_REPORT.md` — detailed report of the smoke run, what worked, what didn't, and time estimates for scaling up

```
.
├── README.md                          # this file
├── requirements.txt
├── src/
│   ├── abstract.py                    # V_abs token defs, constrained-decode LogitsProcessor
│   ├── data_utils.py                  # bottleneck/distill sequence builders, 4-D block attention mask
│   ├── extend_model.py                # tokenizer + embedding extension (M=64 V_abs + 2 delimiters)
│   ├── train_phase_lora.py            # one SFT phase (bottleneck or distill) — DDP + LoRA
│   ├── train_phase.py                 # same, DeepSpeed ZeRO-3 path (full-FT; see caveats)
│   ├── gen_traces.py                  # on-policy abstract-trace generation (HF generate)
│   ├── merge_lora.py                  # merge peft adapter + base into a full HF checkpoint
│   └── eval_math.py                   # vLLM MATH-500 eval (baseline + abstract modes)
├── scripts/
│   ├── setup.sh                       # install python deps, check GPU health
│   ├── check_gpus.sh                  # detect GPUs that pass ECC + a live kernel launch
│   ├── download_data.py               # MATH-500 + filtered Dolci-Think-SFT subset
│   ├── 01_extend_model.sh
│   ├── 02_baseline_eval.sh
│   ├── 03_phase_a.sh
│   ├── 04_merge_lora.sh
│   ├── 05_gen_traces.sh
│   ├── 06_phase_b.sh
│   ├── 07_eval_warmup.sh
│   └── run_smoke.sh                   # one-shot driver: 1-3 above in order
├── configs/
│   ├── accelerate_ds.yaml             # accelerate launcher (DeepSpeed backend)
│   └── ds_zero3.json                  # ZeRO-3 config (for full-FT path)
├── data/                              # populated by scripts/download_data.py
├── runs/                              # populated by training/eval scripts
└── docs/
    └── SMOKE_REPORT.md
```

---

## Reproducing the smoke result on a fresh machine

### 0. Prerequisites

- Linux box with **NVIDIA GPUs**. Smoke was verified on 2-3× A100-SXM4-40GB; anything with ≥80 GB total VRAM should work for LoRA. Paper used 8× H100-80GB.
- **PyTorch with matching CUDA** already installed (this repo does not pin torch — match it to your driver). Verified on `torch 2.11.0+cu130` / Python 3.14 / Ubuntu 24.04.
- **HF Hub access** (the model and datasets are public; no token needed, but set `HF_TOKEN` if you have one for faster rate limits).

### 1. Clone + setup

```bash
git clone <this-repo> ThinkingWithoutWordsRepro
cd ThinkingWithoutWordsRepro
bash scripts/setup.sh
```

`setup.sh` installs `requirements.txt`, runs an import sanity check, and prints the GPU indices that pass an ECC + live-kernel check. Note the line:

```
Usable GPU indices: 0,1,2
  Suggested: export CUDA_VISIBLE_DEVICES=0,1,2
```

If `check_gpus.sh` finds 0 usable GPUs, training will fail — see [GPU triage](#gpu-triage-known-issues) below.

### 2. One-shot smoke run

```bash
export CUDA_VISIBLE_DEVICES=0,1,2          # or whatever check_gpus.sh suggested
bash scripts/run_smoke.sh
```

This runs the full pipeline end-to-end:

1. Downloads MATH-500 and a 5k subset of `allenai/Dolci-Think-SFT-7B` (~1 min).
2. Extends Qwen3-4B with the abstract vocab (~30 s + a 7.6 GB checkpoint to disk).
3. Calibrates the baseline on MATH-500 (~1 min, **target ~83% / ~1067 tokens**).
4. Runs one PI round: Phase A (bottleneck SFT) → merge → on-policy traces → Phase B (self-distill) → merge.
5. Evaluates the warm-up model on MATH-500.

Outputs land in `runs/`:

- `runs/baseline_math500.jsonl` — per-example baseline results
- `runs/qwen3-4b-abs/pi1_phaseB_merged/` — final warm-up checkpoint
- `runs/abstract_math500_T1_N5000.jsonl` — per-example warm-up results

Expected wall (on the configuration used to author this repo: 2× A100-40GB):

| Stage | Wall |
|---|---|
| Setup + downloads | ~5 min |
| Extend model | ~30 s |
| Baseline eval | ~1.5 min |
| Phase A (5k, 1ep) | ~33 min |
| `gen_traces` (5k) | ~11 min |
| Phase B (5k, 1ep) | ~16 min |
| Merges (×2) | ~1 min |
| Warm-up eval | ~1 min |
| **Total** | **~70 min** |

On 3 GPUs, training drops to ~22 min for Phase A / ~10 min for Phase B (linear DDP scaling).

### 3. Expected results (smoke)

| Method | MATH-500 acc | Mean tokens |
|---|---|---|
| Paper Baseline (Qwen3-4B, verbal CoT) | 83.2 | 1087 |
| **This repo's Baseline** | **~83.6** | **~1067** |
| Paper Abstract-CoT (Warm-up) | 86.2 | 168 |
| **This repo's smoke Warm-up** | **~73.2** | **~433** |

Baseline matches paper to within noise. **Warm-up underperforms** the paper because the smoke takes shortcuts (T=1 instead of 3, 5k instead of 600k, LoRA instead of full FT, 1 epoch instead of 3). See [`docs/SMOKE_REPORT.md`](docs/SMOKE_REPORT.md) for a complete breakdown of where the 10-point gap comes from and how to close it.

---

## Running pieces independently

Each script is parametrised by env vars. Defaults match the smoke run.

```bash
# Get the data
python3 scripts/download_data.py --n 5000 --dolci-out data/dolci_5k.jsonl

# Extend Qwen3-4B
bash scripts/01_extend_model.sh

# Baseline calibration
bash scripts/02_baseline_eval.sh

# Phase A (random Z̃, t=1)
BASE=runs/qwen3-4b-abs/base OUT=runs/qwen3-4b-abs/pi1_phaseA \
  bash scripts/03_phase_a.sh

# Merge Phase A
BASE=runs/qwen3-4b-abs/base ADAPTER=runs/qwen3-4b-abs/pi1_phaseA \
  OUT=runs/qwen3-4b-abs/pi1_phaseA_merged \
  bash scripts/04_merge_lora.sh

# On-policy traces for self-distill (use_cot=false, conditioned on x only)
BASE=runs/qwen3-4b-abs/pi1_phaseA_merged \
  OUT=runs/qwen3-4b-abs/traces_distill.jsonl \
  USE_COT=false \
  bash scripts/05_gen_traces.sh

# Phase B
BASE=runs/qwen3-4b-abs/pi1_phaseA_merged \
  TRACES_FILE=runs/qwen3-4b-abs/traces_distill.jsonl \
  OUT=runs/qwen3-4b-abs/pi1_phaseB \
  bash scripts/06_phase_b.sh

# Merge Phase B
BASE=runs/qwen3-4b-abs/pi1_phaseA_merged ADAPTER=runs/qwen3-4b-abs/pi1_phaseB \
  OUT=runs/qwen3-4b-abs/pi1_phaseB_merged \
  bash scripts/04_merge_lora.sh

# Eval warm-up
MODEL=runs/qwen3-4b-abs/pi1_phaseB_merged \
  OUT=runs/abstract_math500.jsonl \
  bash scripts/07_eval_warmup.sh
```

### Scaling up

To run at "mid-fidelity" (60k × T=3 × 3 epochs, LoRA — projected ~2.5 days on 3× A100-40GB):

```bash
python3 scripts/download_data.py --n 60000 --dolci-out data/dolci_60k.jsonl
N=60000 EPOCHS=3 T=3 bash scripts/run_smoke.sh
```

For the full-FT path (paper-faithful but slower): use `src/train_phase.py` with `configs/accelerate_ds.yaml`. **Caveat:** DeepSpeed's CPU-Adam offload may fail to build on torch 2.11+cu13 — see [`docs/SMOKE_REPORT.md`](docs/SMOKE_REPORT.md) §1. Without offload, memory at seq_len > 1024 will likely OOM on 40 GB cards.

---

## Method, in one screen

The paper's Abstract-CoT replaces a verbal chain-of-thought with a short sequence of tokens from a **reserved abstract vocabulary** `V_abs` (M=64). The model is trained to use this short discrete latent trace as a "reasoning scratchpad" before emitting its answer.

```
prompt  ─►  <beginabstract>  z_1 ... z_m  <endabstract>  answer
            └─────── z̃ ∈ V_abs^m ───────┘
```

Training proceeds in two phases per PI round, repeated T=3 times in the paper:

1. **Phase A — Bottlenecked SFT.** Train on `[X; C; Z̃; Y]` with a custom block attention mask: the answer Y is **forbidden from attending to the verbal CoT C**, so all information from C → Y must flow through Z̃. At t=1, Z̃ is random V_abs sequences; at t≥2, Z̃ is sampled on-policy from the previous round's model via constrained decoding.
2. **Phase B — Self-distillation.** Train on `[X; Z̃; Y]` with standard causal masking, where Z̃ is now generated from `x` alone (no CoT) via constrained decoding. This adapts the model to producing the abstract trace from the prompt directly.

After T rounds of warm-up, the paper does RL (GRPO with KL to the warm-started reference) to refine the abstract policy. This repo stops at the end of warm-up.

The most important implementation details (and their gotchas) are documented inline in the source files. The 4-D block attention mask is in `src/data_utils.py`; the constrained-decoding logits processor is in `src/abstract.py`.

---

## GPU triage (known issues)

`scripts/check_gpus.sh` filters out GPUs that fail either (a) recent volatile uncorrected ECC errors, or (b) a live bf16 matmul kernel launch. **High *aggregate* ECC counts are ignored** — they reflect history, not current state, and we observed at least one GPU (high aggregate, zero volatile after a reset) pass the live test and serve correctly.

If you see `cudaErrorECCUncorrectable` mid-run, treat the affected GPU as dead and re-run `check_gpus.sh`.

### vLLM tensor-parallel size

Qwen3-4B has 32 attention heads. vLLM requires `tensor_parallel_size` to divide that → only `TP ∈ {1, 2, 4, 8, 16}`. With 3 GPUs you must use `TP=2` (one GPU idle during eval) or `TP=1` with 3 replicas. Training (DDP) doesn't have this constraint — `accelerate launch --num_processes 3` is fine.

### LoRA + tied embeddings

Qwen3-4B sets `tie_word_embeddings=True`. peft's `modules_to_save=["embed_tokens", "lm_head"]` silently untied them during our run (config flipped to `False` at merge time). This works, but if you're scaling up, consider passing `ensure_weight_tying=True` to `LoraConfig` to be explicit.

---

## Citing the original paper

```
@article{ramji2026thinking,
  title={Thinking Without Words: Efficient Latent Reasoning with Abstract Chain-of-Thought},
  author={Ramji, Keshav and Naseem, Tahira and Fernandez Astudillo, Ramón},
  journal={arXiv preprint arXiv:2604.22709},
  year={2026}
}
```
