"""Generate on-policy abstract traces from a single-GPU loaded model.

Used between training phases (no DeepSpeed/ZeRO complications).
"""
from __future__ import annotations
import argparse, json, random, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

from abstract import BEGIN_ABS, END_ABS, abstract_token_strings, AbstractConstrainedLogits
from data_utils import encode_user_prefix, load_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Trained model dir")
    ap.add_argument("--data", default="/workspace/data/dolci_5k.jsonl")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--use-cot", action="store_true", help="If set, condition on x+c (for bottleneck t>=2). Else x only (distill).")
    ap.add_argument("--m-max", type=int, default=128)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-prefix-len", type=int, default=1024)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    abs_token_ids = tok.convert_tokens_to_ids(abstract_token_strings(64))
    begin_id = tok.convert_tokens_to_ids(BEGIN_ABS)
    end_id = tok.convert_tokens_to_ids(END_ABS)

    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="sdpa", device_map="cuda:0",
    )
    model.eval()

    rows = load_jsonl(args.data)[: args.n]
    proc = AbstractConstrainedLogits(abs_token_ids=abs_token_ids, end_id=end_id, m_max=args.m_max, begin_id=begin_id)
    plist = LogitsProcessorList([proc])

    # Build prompts
    prompts = []
    for r in rows:
        X = encode_user_prefix(tok, r["prompt"])
        if args.use_cot:
            C = tok(r["cot"], add_special_tokens=False).input_ids
        else:
            C = []
        prefix = X + C + [begin_id]
        if len(prefix) > args.max_prefix_len:
            keep = args.max_prefix_len - len(X) - 1
            C = C[:max(0, keep)]
            prefix = X + C + [begin_id]
        prompts.append(prefix)

    # Order by length, batch
    abs_set = set(int(x) for x in abs_token_ids)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    out_traces = [None] * len(prompts)

    pad_id = tok.pad_token_id
    t0 = time.time()
    i = 0
    while i < len(order):
        chunk_idx = order[i:i + args.batch]
        chunk = [prompts[j] for j in chunk_idx]
        L = max(len(p) for p in chunk)
        ids = torch.full((len(chunk), L), pad_id, dtype=torch.long, device="cuda:0")
        attn = torch.zeros((len(chunk), L), dtype=torch.long, device="cuda:0")
        for k, p in enumerate(chunk):
            ids[k, L - len(p):] = torch.tensor(p, dtype=torch.long)
            attn[k, L - len(p):] = 1
        with torch.no_grad():
            gen = model.generate(
                input_ids=ids, attention_mask=attn,
                do_sample=True, temperature=1.0, top_p=1.0,
                max_new_tokens=args.m_max + 2,
                pad_token_id=pad_id, eos_token_id=end_id,
                logits_processor=plist,
            )
        for k, j in enumerate(chunk_idx):
            new = gen[k, L:].tolist()
            if end_id in new:
                new = new[: new.index(end_id)]
            trace = [t for t in new if t in abs_set]
            if not trace:
                trace = [random.choice(abs_token_ids) for _ in range(8)]
            out_traces[j] = trace
        i += args.batch
        if (i // args.batch) % 20 == 0:
            done = min(i, len(order))
            rate = done / max(1, int(time.time()-t0))
            print(f"  gen: {done}/{len(prompts)}  rate={rate:.1f}/s  elapsed={int(time.time()-t0)}s")

    # Save in original row order
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        lens = []
        for trace in out_traces:
            f.write(json.dumps({"trace": trace}) + "\n")
            lens.append(len(trace))
    lens.sort()
    n = len(lens)
    print(f"DONE wrote {n} traces to {args.out}  t={int(time.time()-t0)}s")
    print(f"  trace lens: mean={sum(lens)/n:.1f}  median={lens[n//2]}  p10={lens[n//10]}  p90={lens[int(0.9*n)]}  max={max(lens)}")


if __name__ == "__main__":
    main()
