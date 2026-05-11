#!/usr/bin/env bash
# One-time environment setup: system deps + python deps.
# Re-runnable; safe to invoke after pulling new code.
#
# Assumes:
#   - You have a working CUDA + matching torch already installed (this script
#     does NOT install torch — pin yours via the base image).
#   - You're running as a user that can `apt-get install` (or skip and install
#     poppler-utils manually).
#
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$HERE/.." && pwd)"
cd "$REPO"

echo "== Python =="
python3 --version
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'devices', torch.cuda.device_count())"

echo "== System deps =="
# pdftotext / pdftoppm — only needed if you want to ingest the paper PDF locally.
if ! command -v pdftotext >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get install -y poppler-utils >/dev/null
  fi
fi

echo "== Python deps =="
pip install --quiet -r requirements.txt

echo "== Sanity =="
python3 -c "
import transformers, accelerate, datasets, peft, vllm, math_verify
print('transformers', transformers.__version__)
print('accelerate  ', accelerate.__version__)
print('datasets    ', datasets.__version__)
print('peft        ', peft.__version__)
print('vllm        ', vllm.__version__)
print('math_verify OK')
"

echo "== GPU health =="
USABLE="$(bash scripts/check_gpus.sh)"
if [[ -z "$USABLE" ]]; then
  echo "WARNING: no usable GPUs detected (or nvidia-smi not present). Training will fail."
else
  echo "Usable GPU indices: $USABLE"
  echo "  Suggested: export CUDA_VISIBLE_DEVICES=$USABLE"
fi

echo "Setup complete."
