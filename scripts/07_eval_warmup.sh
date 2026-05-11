#!/usr/bin/env bash
# Eval an Abstract-CoT warm-up model on MATH-500.
# Two-stage generation: constrained abstract-vocab decode up to m_max tokens
# (terminate on <endabstract> or m_max), then unconstrained answer decode.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$HERE/.." && pwd)"
cd "$REPO"

: "${MODEL:?MODEL = merged warm-up model dir (e.g. runs/qwen3-4b-abs/pi3_phaseB_merged)}"
: "${OUT:?OUT = output jsonl path for per-example results}"
: "${DATA_DIR:=$REPO/data}"
: "${HF_HOME:=$REPO/cache}"
: "${TP:=2}"                # vLLM TP must divide num_attention_heads
: "${MAX_NEW:=8192}"
: "${M_MAX:=128}"
: "${M_MIN:=16}"            # 0 to disable forced min length
: "${ABS_TEMP:=0.7}"
: "${SEED:=42}"
export HF_HOME

echo "Abstract eval: model=$MODEL  tp=$TP  m_min=$M_MIN  m_max=$M_MAX  abs_temp=$ABS_TEMP"
python3 src/eval_math.py \
  --model "$MODEL" \
  --data "$DATA_DIR/math500.jsonl" \
  --mode abstract \
  --out "$OUT" \
  --tp "$TP" \
  --max-new-tokens "$MAX_NEW" \
  --m-max "$M_MAX" --m-min "$M_MIN" \
  --abs-temp "$ABS_TEMP" --seed "$SEED"
