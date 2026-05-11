"""Download MATH-500 and a filtered subset of Dolci-Think-SFT-7B for warm-up training.

Defaults match what the smoke run used (5k rows, math+python heavy).

Usage:
  python scripts/download_data.py --n 5000 --out data/dolci_5k.jsonl
  python scripts/download_data.py --n 60000 --out data/dolci_60k.jsonl

The filter keeps assistant messages that contain a <think>...</think> block
with at least --min-cot-chars characters. With the default `buffer-size`, the
streaming shuffle skips past the early Tulu-Wildchat parquet shards (which have
mostly-empty <think> blocks) into the OpenThoughts3 / SYNTHETIC-2 shards.

If you set --sources to a comma-separated allowlist of dataset_source prefixes,
only matching examples are kept (e.g. --sources OpenThoughts3,SYNTHETIC).
"""
from __future__ import annotations
import argparse, json, os, random, re, sys, time
from pathlib import Path

import datasets


TH = re.compile(r"<think>(.*?)</think>", re.S)


def download_math500(out_path: str):
    print(f"[math500] downloading HuggingFaceH4/MATH-500 -> {out_path}")
    ds = datasets.load_dataset("HuggingFaceH4/MATH-500", split="test")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    ds.to_json(out_path)
    print(f"[math500] {len(ds)} problems written")


def download_dolci(out_path: str, n: int, min_cot_chars: int, buffer_size: int,
                   sources: list[str] | None, seed: int, dataset: str):
    print(f"[dolci] streaming {dataset} ({sources=}, target {n} rows)")
    ds = datasets.load_dataset(dataset, split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=buffer_size)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    kept, seen = 0, 0
    by_src: dict[str, int] = {}
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
                print(f"  kept={kept}/{n}  seen={seen}  t={int(time.time()-t0)}s")
            if kept >= n:
                break
    print(f"[dolci] done: kept={kept} seen={seen} t={int(time.time()-t0)}s")
    print("[dolci] sources:")
    for s, c in sorted(by_src.items(), key=lambda x: -x[1])[:15]:
        print(f"  {c:6d}  {s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--math500-out", default="data/math500.jsonl")
    ap.add_argument("--dolci-out", default="data/dolci_5k.jsonl")
    ap.add_argument("--n", type=int, default=5000, help="number of dolci rows to keep")
    ap.add_argument("--dolci-dataset", default="allenai/Dolci-Think-SFT-7B")
    ap.add_argument("--min-cot-chars", type=int, default=200)
    ap.add_argument("--buffer-size", type=int, default=20000)
    ap.add_argument("--sources", default="", help="comma-separated substring filter for dataset_source; empty = no filter")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-math500", action="store_true")
    ap.add_argument("--skip-dolci", action="store_true")
    args = ap.parse_args()

    sources = [s for s in args.sources.split(",") if s.strip()] or None

    if not args.skip_math500:
        download_math500(args.math500_out)
    if not args.skip_dolci:
        download_dolci(args.dolci_out, args.n, args.min_cot_chars,
                       args.buffer_size, sources, args.seed, args.dolci_dataset)


if __name__ == "__main__":
    main()
