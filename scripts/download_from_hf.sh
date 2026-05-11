#!/usr/bin/env bash
# Download the trained Abstract-CoT warm-up artifacts from HuggingFace on a fresh machine.
#
# Usage:
#   bash scripts/download_from_hf.sh                 # default: download final/ + results + docs (~8.5 GB)
#   bash scripts/download_from_hf.sh --full          # everything (~35 GB)
#   bash scripts/download_from_hf.sh --adapters-only # only LoRA adapters (~11 GB) — needs your own extended base
#
# Prereqs:
#   pip install -U "huggingface_hub[cli]"
#   hf auth login   (only needed if the repo becomes private; public access works without it)
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$HERE/.." && pwd)"
cd "$REPO"

: "${HF_REPO:=leapeto/Qwen3-4B-AbstractCoT-warmup}"
: "${OUT_DIR:=$REPO/runs_hf}"
mkdir -p "$OUT_DIR"

MODE="${1:-default}"

case "$MODE" in
  --full)
    echo ">> Downloading EVERYTHING (~35 GB) from $HF_REPO -> $OUT_DIR"
    hf download "$HF_REPO" --local-dir "$OUT_DIR"
    ;;
  --adapters-only)
    echo ">> Downloading LoRA adapters + logs + results (~11 GB) from $HF_REPO -> $OUT_DIR"
    hf download "$HF_REPO" \
      --include "adapters/**" "train_logs/**" "results/**" "docs/**" "README.md" \
      --local-dir "$OUT_DIR"
    ;;
  --intermediates)
    echo ">> Downloading final + round1 + round2 (~25 GB) from $HF_REPO -> $OUT_DIR"
    hf download "$HF_REPO" \
      --include "final/**" "round1/**" "round2/**" "results/**" "train_logs/**" "docs/**" "README.md" \
      --local-dir "$OUT_DIR"
    ;;
  default|*)
    echo ">> Downloading final model + results + logs + docs (~8.5 GB) from $HF_REPO -> $OUT_DIR"
    hf download "$HF_REPO" \
      --include "final/**" "results/**" "train_logs/**" "docs/**" "README.md" \
      --local-dir "$OUT_DIR"
    ;;
esac

echo ""
echo "Done. Final warm-up model is at: $OUT_DIR/final/"
echo ""
echo "Quick smoke test (vLLM, TP=2):"
echo "  python3 src/eval_math.py \\"
echo "    --model $OUT_DIR/final \\"
echo "    --data data/math500.jsonl \\"
echo "    --mode abstract \\"
echo "    --out $OUT_DIR/abstract_math500.jsonl \\"
echo "    --tp 2 --max-new-tokens 8192 --m-max 128 --m-min 16 --abs-temp 0.7"
