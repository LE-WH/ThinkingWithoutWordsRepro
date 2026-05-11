"""One SFT phase under DeepSpeed ZeRO-3 (called separately for bottleneck vs distill).

Usage:
  accelerate launch --config_file accelerate_ds.yaml train_phase.py \
      --base BASE_DIR --mode bottleneck --traces-file traces.json --out OUT
"""
from __future__ import annotations
import argparse, json, os, random, time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
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
        # Trace must be a list of ints
        if self.mode == "bottleneck":
            ex = build_bottleneck_example(self.tok, r["prompt"], r["cot"], r["answer"], trace,
                                          self.begin_id, self.end_id, self.max_len)
        else:
            ex = build_distill_example(self.tok, r["prompt"], r["answer"], trace,
                                       self.begin_id, self.end_id, self.max_len)
        if ex is None:
            # Drop heavy cot to fit
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
    args = ap.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)

    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum, mixed_precision="bf16")
    if accelerator.is_main_process:
        print("State:", accelerator.state)
        print("World:", accelerator.num_processes)

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    abs_token_ids = tok.convert_tokens_to_ids(abstract_token_strings(64))
    begin_id = tok.convert_tokens_to_ids(BEGIN_ABS)
    end_id = tok.convert_tokens_to_ids(END_ABS)

    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="sdpa",
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
    from functools import partial
    collate_fn = partial(collate, pad_id=pad_id)
    dl = DataLoader(ds, batch_size=args.micro_batch, shuffle=True, collate_fn=collate_fn,
                    num_workers=0, pin_memory=True, drop_last=True)

    steps_per_epoch = max(1, len(dl) // args.grad_accum)
    total_steps = steps_per_epoch * args.epochs

    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr,
                betas=(0.9, 0.95), weight_decay=0.0, eps=1e-8)
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=max(1, total_steps // 20),
                                            num_training_steps=total_steps)

    model, opt, dl, sched = accelerator.prepare(model, opt, dl, sched)
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
                            print(f"[{args.mode}] ep{ep} step {step}/{total_steps} loss={avg:.4f} lr={sched.get_last_lr()[0]:.2e} t={int(time.time()-t0)}s")
                        accum_loss, accum_n = 0.0, 0
        if accelerator.is_main_process:
            print(f"[{args.mode}] epoch {ep+1}/{args.epochs} done in {int(time.time()-t0)}s")

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
                       "n_examples": len(rows), "epochs": args.epochs, "mode": args.mode}, f, indent=2)
        print(f"[{args.mode}] saved to {args.out} in {int(time.time()-t0)}s")


if __name__ == "__main__":
    main()
