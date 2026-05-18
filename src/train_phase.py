"""One SFT phase under DeepSpeed ZeRO-3 (called separately for bottleneck vs distill).

Usage:
  accelerate launch --config_file accelerate_ds.yaml train_phase.py \
      --base BASE_DIR --mode bottleneck --traces-file traces.json --out OUT
"""
from __future__ import annotations
import argparse, contextlib, datetime, json, os, random, time
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs

from abstract import BEGIN_ABS, END_ABS, abstract_token_strings
from data_utils import (IGNORE, load_jsonl, random_abstract_trace,
                        build_bottleneck_pass1, build_bottleneck_pass2,
                        build_distill_example, collate, collate_twopass)


def _best_attn_impl() -> str:
    try:
        import flash_attn  # noqa: F401
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
    ap.add_argument("--data", default="/workspace/data/dolci_5k.jsonl")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--mode", choices=["bottleneck", "distill"], required=True)
    ap.add_argument("--traces-file", default=None, help="Optional jsonl with field 'trace' (list of int) per row. Required for t>=2 bottleneck and for distill.")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-every", type=int, default=0,
                    help="save checkpoint every N optimizer steps (0 = only at end)")
    # W&B
    ap.add_argument("--wandb-project", default="", help="W&B project; empty = no W&B logging")
    ap.add_argument("--wandb-run-name", default="", help="W&B run name; empty = auto")
    args = ap.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)

    pg_kwargs = InitProcessGroupKwargs(timeout=datetime.timedelta(hours=2))
    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum, mixed_precision="bf16",
                              kwargs_handlers=[pg_kwargs])
    if accelerator.is_main_process:
        print("State:", accelerator.state)
        print("World:", accelerator.num_processes)

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    abs_token_ids = tok.convert_tokens_to_ids(abstract_token_strings(64))
    begin_id = tok.convert_tokens_to_ids(BEGIN_ABS)
    end_id = tok.convert_tokens_to_ids(END_ABS)

    attn_impl = _best_attn_impl()
    if accelerator.is_main_process:
        print(f"attn_implementation: {attn_impl}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, trust_remote_code=True, attn_implementation=attn_impl,
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    rows = load_jsonl(args.data)[: args.n]

    traces = None
    if args.traces_file:
        traces = []
        with open(args.traces_file) as f:
            for ln in f:
                d = json.loads(ln)
                traces.append(d["trace"])
        assert len(traces) == len(rows), f"trace count {len(traces)} != rows {len(rows)}"
        if accelerator.is_main_process:
            print(f"loaded {len(traces)} pre-generated traces")

    ds = WarmupDataset(rows, tok, args.mode, traces, abs_token_ids, begin_id, end_id, args.max_len)
    pad_id = tok.pad_token_id
    collate_fn = (partial(collate_twopass, pad_id=pad_id) if args.mode == "bottleneck"
                  else partial(collate, pad_id=pad_id))
    dl = DataLoader(ds, batch_size=args.micro_batch, shuffle=True, collate_fn=collate_fn,
                    num_workers=0, pin_memory=True, drop_last=True)

    steps_per_epoch = max(1, len(dl) // args.grad_accum)
    total_opt_steps = steps_per_epoch * args.epochs
    # AcceleratedScheduler calls underlying sched.step() num_processes times per
    # optimizer step — pre-multiply total_steps so cosine doesn't overshoot.
    total_steps = total_opt_steps * accelerator.num_processes
    warmup = max(1, total_steps // 20)

    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr,
                betas=(0.9, 0.95), weight_decay=0.0, eps=1e-8)
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=warmup,
                                            num_training_steps=total_steps)

    model, opt, dl, sched = accelerator.prepare(model, opt, dl, sched)

    # W&B init (main process only)
    try:
        import wandb as _wandb
    except ImportError:
        _wandb = None
    _wb = None
    if args.wandb_project and accelerator.is_main_process:
        if _wandb is None:
            print("WARNING: wandb not installed, skipping W&B logging")
        else:
            _wb = _wandb
            _wb.init(project=args.wandb_project, name=args.wandb_run_name or None,
                     config={"mode": args.mode, "base": args.base, "n": args.n,
                             "epochs": args.epochs, "micro_batch": args.micro_batch,
                             "grad_accum": args.grad_accum, "lr": args.lr,
                             "max_len": args.max_len, "attn_impl": attn_impl,
                             "n_gpus": accelerator.num_processes})

    model.train()

    t0 = time.time()
    step = 0
    accum_loss, accum_loss_z, accum_loss_y, accum_n = 0.0, 0.0, 0.0, 0
    losses = []
    for ep in range(args.epochs):
        for batch in dl:
            with accelerator.accumulate(model):
                if args.mode == "bottleneck":
                    batch1, batch2 = batch
                    # Two-pass: guard pass-1 under no_sync so DDP all-reduce
                    # fires exactly once (after pass-2 backward).
                    _ns = (model.no_sync()
                           if (accelerator.sync_gradients and accelerator.num_processes > 1)
                           else contextlib.nullcontext())
                    loss1 = model(input_ids=batch1["input_ids"],
                                  attention_mask=batch1["attention_mask"],
                                  labels=batch1["labels"], use_cache=False).loss
                    with _ns:
                        accelerator.backward(loss1 / 2)
                    loss2 = model(input_ids=batch2["input_ids"],
                                  attention_mask=batch2["attention_mask"],
                                  labels=batch2["labels"], use_cache=False).loss
                    accelerator.backward(loss2 / 2)
                    accum_loss_z += float(loss1.detach())
                    accum_loss_y += float(loss2.detach())
                    loss = (loss1.detach() + loss2.detach()) / 2
                else:
                    loss = model(input_ids=batch["input_ids"],
                                 attention_mask=batch["attention_mask"],
                                 labels=batch["labels"], use_cache=False).loss
                    accelerator.backward(loss)
                accum_loss += float(loss.detach())
                accum_n += 1

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                    step += 1
                    if step % args.log_every == 0:
                        n   = max(1, accum_n)
                        avg = accum_loss   / n
                        avg_z = accum_loss_z / n
                        avg_y = accum_loss_y / n
                        lr_now = float(sched.get_last_lr()[0])
                        losses.append(avg)
                        if accelerator.is_main_process:
                            if args.mode == "bottleneck":
                                print(f"[bottleneck] ep{ep} step {step}/{total_opt_steps} "
                                      f"loss={avg:.4f} (z={avg_z:.4f} y={avg_y:.4f}) "
                                      f"lr={lr_now:.2e} t={int(time.time()-t0)}s", flush=True)
                            else:
                                print(f"[distill] ep{ep} step {step}/{total_opt_steps} "
                                      f"loss={avg:.4f} lr={lr_now:.2e} t={int(time.time()-t0)}s",
                                      flush=True)
                            if _wb is not None:
                                log = {"train/loss": avg, "train/lr": lr_now}
                                if args.mode == "bottleneck":
                                    log["train/loss_z"] = avg_z
                                    log["train/loss_y"] = avg_y
                                _wb.log(log, step=step)
                        accum_loss = accum_loss_z = accum_loss_y = accum_n = 0
                        t0 = time.time()

                    if args.save_every > 0 and step % args.save_every == 0:
                        ckpt = f"{args.out}_step{step:05d}"
                        if accelerator.is_main_process:
                            Path(ckpt).mkdir(parents=True, exist_ok=True)
                            unwrap_ckpt = accelerator.unwrap_model(model)
                            unwrap_ckpt.save_pretrained(
                                ckpt, save_function=accelerator.save,
                                safe_serialization=True,
                                state_dict=accelerator.get_state_dict(model),
                            )
                            tok.save_pretrained(ckpt)
                            print(f"[ckpt] step={step} → {ckpt}", flush=True)
                        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            print(f"[{args.mode}] epoch {ep+1}/{args.epochs} done in {int(time.time()-t0)}s",
                  flush=True)

    accelerator.wait_for_everyone()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    unwrap = accelerator.unwrap_model(model)
    unwrap.save_pretrained(
        args.out,
        save_function=accelerator.save,
        safe_serialization=True,
        state_dict=accelerator.get_state_dict(model),
    )
    if accelerator.is_main_process:
        tok.save_pretrained(args.out)
        with open(os.path.join(args.out, "train_log.json"), "w") as f:
            json.dump({"losses": losses, "wallclock_s": int(time.time()-t0),
                       "n_examples": len(rows), "epochs": args.epochs, "mode": args.mode,
                       "total_opt_steps": total_opt_steps,
                       "num_processes": accelerator.num_processes}, f, indent=2)
        print(f"[{args.mode}] saved to {args.out} in {int(time.time()-t0)}s", flush=True)
        if _wb is not None:
            _wb.finish()


if __name__ == "__main__":
    main()
