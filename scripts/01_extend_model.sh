#!/usr/bin/env bash
# Extend Qwen3-4B tokenizer with the abstract vocab (M=64 + 2 delimiters) and
# resize embeddings. Saves a self-contained base checkpoint to RUNS_DIR/base.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$HERE/.." && pwd)"
cd "$REPO"

: "${BASE_MODEL:=Qwen/Qwen3-4B}"
: "${RUNS_DIR:=$REPO/runs}"
: "${HF_HOME:=$REPO/cache}"
: "${M:=64}"
export HF_HOME

mkdir -p "$RUNS_DIR" "$HF_HOME"
OUT="$RUNS_DIR/qwen3-4b-abs/base"

echo "Extending $BASE_MODEL with M=$M abstract tokens -> $OUT"
python3 src/extend_model.py --src "$BASE_MODEL" --out "$OUT" --M "$M" --init mean
