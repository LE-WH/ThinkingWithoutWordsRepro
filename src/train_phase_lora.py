"""LoRA-adapted SFT phase, single-script multi-GPU DDP (no DeepSpeed needed).

LoRA on attn (q,k,v,o) + MLP (gate,up,down); full-train embed_tokens + lm_head
so the new abstract-vocab rows can move freely. Adam states drop ~32GB -> ~3GB.

Usage:
  accelerate launch --num_processes 2 --mixed_precision bf16 train_phase_lora.py \
    --base BASE --mode bottleneck --epochs 1 --out OUT [--traces-file traces.json]
"""
from __future__ import annotations
import argparse, json, os, random, time
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator

from abstract import BEGIN_ABS, END_ABS, abstract_token_strings, AbstractConstrainedLogits
from data_utils import (IGNORE, load_jsonl, random_abstract_trace,
                        build_bottleneck_pass1, build_bottleneck_pass2,
                        build_distill_example, collate, collate_twopass)

try:
    import wandb as _wandb
except ImportError:
    _wandb = None


_BOXED_RE = __import__("re").compile(r"\\boxed\{")

def _extract_last_boxed(s: str):
    starts = [m.start() for m in _BOXED_RE.finditer(s)]
    if not starts:
        return None
    i, depth, out = starts[-1] + len("\\boxed{"), 1, []
    while i < len(s) and depth > 0:
        c = s[i]
        if c == "{":
            depth += 1; out.append(c)
        elif c == "}":
            depth -= 1
            if depth > 0: out.append(c)
        else:
            out.append(c)
        i += 1
    return "".join(out)


@torch.no_grad()
def run_eval(model, tok, abs_ids, begin_id, end_id, data_path, n, m_max, device):
    """In-training abstract-CoT eval using HF generate (no vLLM, no merge needed).

    Runs two-stage generation: constrained abstract trace, then unconstrained answer.
    Returns dict: acc, mean_abstract_tokens, mean_response_tokens, mean_total_tokens.
    """
    from math_verify import parse, verify as _mv_verify

    def _check(pred, gold):
        if not pred:
            return False
        try:
            return bool(_mv_verify(parse(f"${gold}$"), parse(f"${pred}$")))
        except Exception:
            return (pred or "").strip() == gold.strip()

    SYS = "Please reason step by step, and put your final answer within \\boxed{}."
    rows = load_jsonl(data_path)[:n]

    prev_use_cache = model.config.use_cache
    model.config.use_cache = True
    model.eval()

    correct, abs_lens, resp_lens = 0, [], []
    nl_ids = tok.encode("\n", add_special_tokens=False)

    for r in rows:
        problem = r.get("problem", r.get("prompt", ""))
        gold    = r["answer"]
        msgs    = [{"role": "system", "content": SYS}, {"role": "user", "content": problem}]
        txt     = tok.apply_chat_template(msgs, tokenize=False,
                                          add_generation_prompt=True, enable_thinking=False)
        x = tok(txt, add_special_tokens=False).input_ids

        # Stage 1: generate abstract trace (constrained to V_abs ∪ {END_ABS})
        inp1 = torch.tensor([x + [begin_id]], dtype=torch.long, device=device)
        lp   = AbstractConstrainedLogits(abs_ids, end_id, m_max=m_max, begin_id=begin_id)
        out1 = model.generate(inp1, max_new_tokens=m_max + 2, do_sample=False,
                              logits_processor=[lp], pad_token_id=tok.pad_token_id,
                              use_cache=True)
        z_new = out1[0][len(x) + 1:]
        ep = (z_new == end_id).nonzero(as_tuple=False)
        if len(ep):
            z_new = z_new[: ep[0].item() + 1]

        # Stage 2: generate answer from [X; BEGIN_ABS; Z; END_ABS; \n]
        pfx2 = x + [begin_id] + z_new.tolist()
        if not len(z_new) or z_new[-1].item() != end_id:
            pfx2 += [end_id]
        pfx2 += nl_ids
        inp2 = torch.tensor([pfx2], dtype=torch.long, device=device)
        out2 = model.generate(inp2, max_new_tokens=2048, do_sample=False,
                              pad_token_id=tok.pad_token_id,
                              eos_token_id=tok.eos_token_id, use_cache=True)

        ans  = tok.decode(out2[0][len(pfx2):], skip_special_tokens=True)
        correct += int(_check(_extract_last_boxed(ans), gold))
        abs_toks = [t for t in z_new.tolist() if t not in (begin_id, end_id)]
        abs_lens.append(len(abs_toks))
        resp_lens.append(max(0, len(out2[0]) - len(pfx2)))

    model.config.use_cache = prev_use_cache
    model.train()
    n_rows = len(rows)
    return {
        "acc":                  correct / n_rows,
        "mean_abstract_tokens": sum(abs_lens)  / n_rows,
        "mean_response_tokens": sum(resp_lens) / n_rows,
        "mean_total_tokens":   (sum(abs_lens) + sum(resp_lens)) / n_rows,
        "n_eval":               n_rows,
    }


def _best_attn_impl() -> str:
    """Use FlashAttention-2 on Ampere+ GPUs if available, otherwise SDPA."""
    try:
        import flash_attn  # noqa: F401
        import torch
        if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
            return "flash_attention_2"
    except ImportError:
        pass
    return "sdpa"


class WarmupDataset(Dataset):
    def __init__(self, rows, tok, mode, traces, abs_token_ids, begin_id, end_id, max_len):
        self.rows = rows; self.tok = tok; self.mode = mode
        self.traces = traces
        self.abs_token_ids = abs_token_ids; self.begin_id = begin_id; self.end_id = end_id
        self.max_len = max_len

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        trace = self.traces[i] if self.traces is not None else None
        if trace is None:
            trace = random_abstract_trace(len(self.abs_token_ids), self.abs_token_ids)
        if self.mode == "bottleneck":
            ex1 = build_bottleneck_pass1(self.tok, r["prompt"], r["cot"], trace,
                                         self.begin_id, self.end_id, self.max_len)
            ex2 = build_bottleneck_pass2(self.tok, r["prompt"], r["answer"], trace,
                                         self.begin_id, self.end_id, self.max_len)
            if ex1 is None or ex2 is None:
                return self.__getitem__((i + 1) % len(self))
            return {"pass1": ex1, "pass2": ex2}
        else:
            ex = build_distill_example(self.tok, r["prompt"], r["answer"], trace,
                                       self.begin_id, self.end_id, self.max_len)
            if ex is None:
                return self.__getitem__((i + 1) % len(self))
            return ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--prev-lora", default=None, help="Optional path to previous LoRA dir to start from (merged into base before re-applying LoRA, OR loaded as adapter to continue training).")
    ap.add_argument("--data", default="/workspace/data/dolci_5k.jsonl")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--mode", choices=["bottleneck", "distill"], required=True)
    ap.add_argument("--traces-file", default=None)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    # W&B
    ap.add_argument("--wandb-project", default="", help="W&B project; empty = no W&B logging")
    ap.add_argument("--wandb-run-name", default="", help="W&B run name; empty = auto")
    # In-training eval
    ap.add_argument("--eval-data", default="", help="path to math500.jsonl; empty = skip eval")
    ap.add_argument("--eval-every", type=int, default=100, help="eval every N optimizer steps")
    ap.add_argument("--eval-n", type=int, default=100, help="number of math500 problems per eval")
    ap.add_argument("--eval-m-max", type=int, default=128, help="abstract trace length cap for eval")
    args = ap.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)

    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum, mixed_precision="bf16")
    if accelerator.is_main_process:
        print(f"State: {accelerator.state}")
        print(f"World: {accelerator.num_processes}")

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    abs_token_ids = tok.convert_tokens_to_ids(abstract_token_strings(64))
    begin_id = tok.convert_tokens_to_ids(BEGIN_ABS)
    end_id = tok.convert_tokens_to_ids(END_ABS)

    attn_impl = _best_attn_impl()
    if accelerator.is_main_process:
        print(f"attn_implementation: {attn_impl}")

    # W&B init (main process only)
    _wb = None
    if args.wandb_project and accelerator.is_main_process:
        if _wandb is None:
            print("WARNING: wandb not installed, skipping W&B logging")
        else:
            _wb = _wandb
            _wb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or None,
                config={
                    "mode": args.mode, "base": args.base,
                    "n": args.n, "epochs": args.epochs,
                    "micro_batch": args.micro_batch, "grad_accum": args.grad_accum,
                    "lr": args.lr, "lora_rank": args.lora_rank,
                    "max_len": args.max_len, "attn_impl": attn_impl,
                    "n_gpus": accelerator.num_processes,
                    "eval_every": args.eval_every, "eval_n": args.eval_n,
                },
            )

    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, trust_remote_code=True, attn_implementation=attn_impl,
    )

    if args.prev_lora:
        # Merge prior LoRA into base before applying a new one (cleaner continuation).
        from peft import PeftModel
        if accelerator.is_main_process:
            print(f"loading prior LoRA from {args.prev_lora}, merging into base...")
        model = PeftModel.from_pretrained(model, args.prev_lora)
        model = model.merge_and_unload()

    # Configure LoRA: rank-32, all linear projections; train embed_tokens & lm_head fully.
    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["embed_tokens", "lm_head"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    if accelerator.is_main_process:
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"trainable params: {n_trainable/1e6:.1f}M / {n_total/1e9:.2f}B ({100*n_trainable/n_total:.2f}%)")

    rows = load_jsonl(args.data)[: args.n]
    traces = None
    if args.traces_file:
        traces = []
        with open(args.traces_file) as f:
            for ln in f:
                traces.append(json.loads(ln)["trace"])
        assert len(traces) == len(rows), f"trace count {len(traces)} != rows {len(rows)}"
        if accelerator.is_main_process:
            print(f"loaded {len(traces)} pre-generated traces")

    ds = WarmupDataset(rows, tok, args.mode, traces, abs_token_ids, begin_id, end_id, args.max_len)
    pad_id = tok.pad_token_id
    collate_fn = (partial(collate_twopass, pad_id=pad_id) if args.mode == "bottleneck"
                  else partial(collate, pad_id=pad_id))
    dl = DataLoader(ds, batch_size=args.micro_batch, shuffle=True, collate_fn=collate_fn,
                    num_workers=0, pin_memory=True, drop_last=True)

    opt = AdamW([p for p in model.parameters() if p.requires_grad],
                lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0, eps=1e-8, fused=True)
    model, opt, dl = accelerator.prepare(model, opt, dl)
    steps_per_epoch = max(1, len(dl) // args.grad_accum)
    total_opt_steps = steps_per_epoch * args.epochs
    # AcceleratedScheduler advances the underlying scheduler num_processes times per
    # sched.step() call (split_batches=False, step_with_optimizer=True). Pre-multiply
    # total_steps to match — otherwise the cosine overshoots and bounces back up to peak.
    total_steps = total_opt_steps * accelerator.num_processes
    warmup = max(1, total_steps // 20)
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=warmup,
                                            num_training_steps=total_steps)
    sched = accelerator.prepare(sched)
    if accelerator.is_main_process:
        print(f"LR schedule: {total_opt_steps} opt steps × {accelerator.num_processes} procs "
              f"= {total_steps} underlying steps (warmup={warmup})", flush=True)
    model.train()

    t0 = time.time()
    step = 0
    accum_loss, accum_loss_z, accum_loss_y, accum_n = 0.0, 0.0, 0.0, 0
    losses, lrs = [], []
    for ep in range(args.epochs):
        for batch in dl:
            with accelerator.accumulate(model):
                if args.mode == "bottleneck":
                    batch1, batch2 = batch
                    loss1 = model(input_ids=batch1["input_ids"],
                                  attention_mask=batch1["attention_mask"],
                                  labels=batch1["labels"],
                                  use_cache=False).loss
                    loss2 = model(input_ids=batch2["input_ids"],
                                  attention_mask=batch2["attention_mask"],
                                  labels=batch2["labels"],
                                  use_cache=False).loss
                    loss = (loss1 + loss2) / 2
                    accum_loss_z += float(loss1.detach())
                    accum_loss_y += float(loss2.detach())
                else:
                    loss = model(input_ids=batch["input_ids"],
                                 attention_mask=batch["attention_mask"],
                                 labels=batch["labels"],
                                 use_cache=False).loss
                accelerator.backward(loss)
                accum_loss += float(loss.detach())
                accum_n += 1

                if accelerator.sync_gradients:
                    grad_norm = float(accelerator.clip_grad_norm_(model.parameters(), 1.0))
                    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                    step += 1

                    if step % args.log_every == 0:
                        n = max(1, accum_n)
                        avg      = accum_loss   / n
                        avg_z    = accum_loss_z / n
                        avg_y    = accum_loss_y / n
                        lr_now   = float(sched.get_last_lr()[0])
                        tok_s    = (args.micro_batch * args.max_len * accum_n
                                    * accelerator.num_processes) / max(1, time.time() - t0)
                        losses.append(avg); lrs.append(lr_now)

                        if accelerator.is_main_process:
                            if args.mode == "bottleneck":
                                print(f"[bottleneck] ep{ep} step {step}/{total_opt_steps} "
                                      f"loss={avg:.4f} (z={avg_z:.4f} y={avg_y:.4f}) "
                                      f"gnorm={grad_norm:.3f} lr={lr_now:.2e} "
                                      f"tok/s={tok_s:.0f} t={int(time.time()-t0)}s", flush=True)
                            else:
                                print(f"[distill] ep{ep} step {step}/{total_opt_steps} "
                                      f"loss={avg:.4f} gnorm={grad_norm:.3f} "
                                      f"lr={lr_now:.2e} t={int(time.time()-t0)}s", flush=True)

                            if _wb is not None:
                                log = {"train/loss": avg, "train/lr": lr_now,
                                       "train/grad_norm": grad_norm, "train/tok_per_sec": tok_s}
                                if args.mode == "bottleneck":
                                    log["train/loss_z"] = avg_z
                                    log["train/loss_y"] = avg_y
                                _wb.log(log, step=step)

                        accum_loss = accum_loss_z = accum_loss_y = accum_n = 0
                        t0 = time.time()  # reset throughput window

                    # In-training eval
                    if (args.eval_data and step % args.eval_every == 0):
                        if accelerator.is_main_process:
                            t_eval = time.time()
                            eval_model = accelerator.unwrap_model(model)
                            metrics = run_eval(
                                eval_model, tok, abs_token_ids, begin_id, end_id,
                                args.eval_data, args.eval_n, args.eval_m_max,
                                accelerator.device,
                            )
                            print(f"[eval] step={step} acc={metrics['acc']*100:.2f}%  "
                                  f"abs={metrics['mean_abstract_tokens']:.1f}  "
                                  f"resp={metrics['mean_response_tokens']:.1f}  "
                                  f"total={metrics['mean_total_tokens']:.1f}  "
                                  f"n={metrics['n_eval']}  t={int(time.time()-t_eval)}s",
                                  flush=True)
                            if _wb is not None:
                                _wb.log({f"eval/{k}": v for k, v in metrics.items()}, step=step)
                        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            print(f"[{args.mode}] epoch {ep+1}/{args.epochs} done", flush=True)

    accelerator.wait_for_everyone()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    # Save: peft saves adapter + modules_to_save (embed_tokens, lm_head full).
    unwrap = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        unwrap.save_pretrained(args.out)
        tok.save_pretrained(args.out)
        with open(os.path.join(args.out, "train_log.json"), "w") as f:
            json.dump({"losses": losses, "lrs": lrs, "wallclock_s": int(time.time()-t0),
                       "n_examples": len(rows), "epochs": args.epochs, "mode": args.mode,
                       "lora_rank": args.lora_rank,
                       "total_opt_steps": total_opt_steps,
                       "num_processes": accelerator.num_processes}, f, indent=2)
        print(f"[{args.mode}] saved to {args.out}", flush=True)
        if _wb is not None:
            _wb.finish()


if __name__ == "__main__":
    main()
