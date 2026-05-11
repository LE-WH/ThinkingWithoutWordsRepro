#!/usr/bin/env bash
# Merge a LoRA adapter into its base model and save the result as a full HF
# checkpoint. Required between Phase A and gen_traces (vLLM/HF generate need a
# full model), and between Phase B and eval.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$HERE/.." && pwd)"
cd "$REPO"

: "${BASE:?BASE must point to the base model dir that the LoRA was trained on}"
: "${ADAPTER:?ADAPTER must point to the LoRA dir to merge}"
: "${OUT:?OUT must be the directory to write the merged model to}"

echo "Merging $ADAPTER into $BASE -> $OUT"
# Pin to one GPU; merge is memory-bound, not compute-bound.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}" \
python3 src/merge_lora.py --base "$BASE" --adapter "$ADAPTER" --out "$OUT"
