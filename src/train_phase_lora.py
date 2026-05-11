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

from abstract import BEGIN_ABS, END_ABS, abstract_token_strings
from data_utils import IGNORE, load_jsonl, random_abstract_trace, build_bottleneck_example, build_distill_example, collate


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
            ex = build_bottleneck_example(self.tok, r["prompt"], r["cot"], r["answer"], trace,
                                          self.begin_id, self.end_id, self.max_len)
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

    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="sdpa",
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
    collate_fn = partial(collate, pad_id=pad_id)
    dl = DataLoader(ds, batch_size=args.micro_batch, shuffle=True, collate_fn=collate_fn,
                    num_workers=0, pin_memory=True, drop_last=True)

    opt = AdamW([p for p in model.parameters() if p.requires_grad],
                lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0, eps=1e-8, fused=True)
    # Compute total optimizer steps after accelerate prepare (which adds DistributedSampler)
    model, opt, dl = accelerator.prepare(model, opt, dl)
    steps_per_epoch = max(1, len(dl) // args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=max(1, total_steps // 20),
                                            num_training_steps=total_steps)
    sched = accelerator.prepare(sched)
    model.train()

    t0 = time.time()
    step, accum_loss, accum_n = 0, 0.0, 0
    losses = []
    for ep in range(args.epochs):
        for batch in dl:
            with accelerator.accumulate(model):
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    use_cache=False,
                )
                loss = out.loss
                accelerator.backward(loss)
                accum_loss += float(loss.detach().item())
                accum_n += 1
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                    step += 1
                    if step % args.log_every == 0:
                        avg = accum_loss / max(1, accum_n)
                        losses.append(avg)
                        if accelerator.is_main_process:
                            print(f"[{args.mode}] ep{ep} step {step}/{total_steps} loss={avg:.4f} lr={sched.get_last_lr()[0]:.2e} t={int(time.time()-t0)}s", flush=True)
                        accum_loss, accum_n = 0.0, 0
        if accelerator.is_main_process:
            print(f"[{args.mode}] epoch {ep+1}/{args.epochs} done in {int(time.time()-t0)}s", flush=True)

    accelerator.wait_for_everyone()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    # Save: peft saves adapter + modules_to_save (embed_tokens, lm_head full).
    unwrap = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        unwrap.save_pretrained(args.out)
        tok.save_pretrained(args.out)
        with open(os.path.join(args.out, "train_log.json"), "w") as f:
            json.dump({"losses": losses, "wallclock_s": int(time.time()-t0),
                       "n_examples": len(rows), "epochs": args.epochs, "mode": args.mode,
                       "lora_rank": args.lora_rank}, f, indent=2)
        print(f"[{args.mode}] saved to {args.out} in {int(time.time()-t0)}s", flush=True)


if __name__ == "__main__":
    main()
