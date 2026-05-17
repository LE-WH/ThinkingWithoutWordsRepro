"""Download MATH-500 and a filtered subset of Dolci-Think-SFT-7B for warm-up training.

Defaults match what the smoke run used (5k rows, math+python heavy).

Usage:
  python scripts/download_data.py --n 5000 --out data/dolci_5k.jsonl
  python scripts/download_data.py --n 60000 --out data/dolci_60k.jsonl
  python scripts/download_data.py --n 300000 --max-total-tokens 16384 --out data/dolci_300k_16k.jsonl

The filter keeps assistant messages that contain a <think>...</think> block
with at least --min-cot-chars characters. With the default `buffer-size`, the
streaming shuffle skips past the early Tulu-Wildchat parquet shards (which have
mostly-empty <think> blocks) into the OpenThoughts3 / SYNTHETIC-2 shards.

If you set --sources to a comma-separated allowlist of dataset_source prefixes,
only matching examples are kept (e.g. --sources OpenThoughts3,SYNTHETIC).

--max-total-tokens: if set, keeps only examples where the estimated Phase A
sequence length (X + C + Z + Y) fits within this limit. The estimate uses
char/3 (conservative; actual Qwen3 tokenization is ~3.5 chars/tok) so actual
token counts should land at ≤ 95% of the limit. Empirically ~44% of Dolci
passes at 16384 (~266k of 600k).
"""
from __future__ import annotations
import argparse, json, os, random, re, sys, time
from pathlib import Path

import datasets


TH = re.compile(r"<think>(.*?)</think>", re.S)

# Conservative overhead for chat template prefix (system + user header) and
# abstract trace segment (<beginabstract> 13-tok trace <endabstract>).
_X_OVERHEAD = 150   # tokens: chat template prefix
_Z_OVERHEAD = 15    # tokens: abstract trace segment (delimiters + 13 abstract tokens)


def _est_tokens(prompt: str, cot: str, answer: str) -> int:
    """Estimate Phase A sequence length without running the tokenizer.

    Uses chars/3 (conservative vs the ~3.5 empirical average) so the estimate
    slightly over-counts, keeping actual sequences safely under the limit.
    """
    return _X_OVERHEAD + len(cot) // 3 + _Z_OVERHEAD + len(answer) // 3


def download_math500(out_path: str):
    print(f"[math500] downloading HuggingFaceH4/MATH-500 -> {out_path}")
    ds = datasets.load_dataset("HuggingFaceH4/MATH-500", split="test")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    ds.to_json(out_path)
    print(f"[math500] {len(ds)} problems written")


def download_dolci(out_path: str, n: int, min_cot_chars: int, buffer_size: int,
                   sources: list[str] | None, seed: int, dataset: str,
                   max_total_tokens: int = 0):
    limit_str = f", max_total_tokens={max_total_tokens}" if max_total_tokens else ""
    print(f"[dolci] streaming {dataset} ({sources=}, target {n} rows{limit_str})")
    ds = datasets.load_dataset(dataset, split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=buffer_size)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    kept, seen, skipped_len = 0, 0, 0
    by_src: dict[str, int] = {}
    est_tok_buckets: list[int] = []   # for length distribution at end
    t0 = time.time()
    with open(out_path, "w") as f:
        for ex in ds:
            seen += 1
            src = ex.get("dataset_source", "?")
            if sources and not any(src.lower().find(s.lower()) >= 0 for s in sources):
                continue
            msgs = ex.get("messages") or []
            if len(msgs) < 2 or msgs[-1].get("role") != "assistant":
                continue
            user = next((m["content"] for m in msgs if m.get("role") == "user"), None)
            assistant = msgs[-1]["content"]
            if not user or not assistant:
                continue
            m = TH.search(assistant)
            if not m:
                continue
            cot = m.group(1).strip()
            ans = TH.sub("", assistant).strip()
            if len(cot) < min_cot_chars or len(ans) < 5:
                continue
            if max_total_tokens:
                est = _est_tokens(user, cot, ans)
                if est > max_total_tokens:
                    skipped_len += 1
                    continue
                est_tok_buckets.append(est)
            f.write(json.dumps({
                "id": ex.get("id"),
                "source": src,
                "prompt": user,
                "cot": cot,
                "answer": ans,
            }) + "\n")
            by_src[src] = by_src.get(src, 0) + 1
            kept += 1
            if kept % max(1, n // 10) == 0:
                print(f"  kept={kept}/{n}  seen={seen}  skipped_len={skipped_len}  t={int(time.time()-t0)}s")
            if kept >= n:
                break
    print(f"[dolci] done: kept={kept} seen={seen} skipped_len={skipped_len} t={int(time.time()-t0)}s")
    print("[dolci] sources:")
    for s, c in sorted(by_src.items(), key=lambda x: -x[1])[:15]:
        print(f"  {c:6d}  {s}")
    if est_tok_buckets:
        est_tok_buckets.sort()
        nb = len(est_tok_buckets)
        p = lambda pct: est_tok_buckets[int(nb * pct / 100)]
        print(f"[dolci] estimated token length (Phase A) of kept rows:")
        print(f"  p10={p(10)}  p25={p(25)}  p50={p(50)}  p75={p(75)}  p90={p(90)}  p99={p(99)}  max={est_tok_buckets[-1]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--math500-out", default="data/math500.jsonl")
    ap.add_argument("--dolci-out", default="data/dolci_5k.jsonl")
    ap.add_argument("--n", type=int, default=5000, help="number of dolci rows to keep")
    ap.add_argument("--dolci-dataset", default="allenai/Dolci-Think-SFT-7B")
    ap.add_argument("--min-cot-chars", type=int, default=200)
    ap.add_argument("--buffer-size", type=int, default=20000)
    ap.add_argument("--sources", default="", help="comma-separated substring filter for dataset_source; empty = no filter")
    ap.add_argument("--max-total-tokens", type=int, default=0,
                    help="if >0, skip examples whose estimated Phase A token length exceeds this. "
                         "Uses chars/3 (conservative estimate). E.g. --max-total-tokens 16384")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-math500", action="store_true")
    ap.add_argument("--skip-dolci", action="store_true")
    args = ap.parse_args()

    sources = [s for s in args.sources.split(",") if s.strip()] or None

    if not args.skip_math500:
        download_math500(args.math500_out)
    if not args.skip_dolci:
        download_dolci(args.dolci_out, args.n, args.min_cot_chars,
                       args.buffer_size, sources, args.seed, args.dolci_dataset,
                       max_total_tokens=args.max_total_tokens)


if __name__ == "__main__":
    main()
