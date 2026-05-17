#!/usr/bin/env bash
# One-time environment setup. Re-runnable; safe to invoke after pulling new code.
#
# What this script installs (in order):
#   1. vLLM 0.14.0  →  pins torch 2.9.1+cu128 + all CUDA 12 runtime libraries
#   2. flash-attn 2.8.3 pre-built wheel  (SM80+ / Ampere only; skipped on older GPUs)
#   3. Training dependencies from requirements.txt
#
# Assumptions:
#   - Python 3.12, Linux x86_64
#   - GPU driver compatible with CUDA 12 (driver >= 525)
#   - Internet access to PyPI and GitHub releases
#
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$HERE/.." && pwd)"
cd "$REPO"

# ── 1. Python version check ───────────────────────────────────────────────────
echo "== Python =="
python3 --version
PY_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
if [[ "$PY_MINOR" != "12" ]]; then
  echo "WARNING: Python 3.12 expected (got 3.${PY_MINOR})."
  echo "  The pre-built flash-attn wheel targets cp312."
  echo "  If flash-attn install fails, it will fall back to source build."
fi

# ── 2. System deps ────────────────────────────────────────────────────────────
echo "== System deps =="
if ! command -v pdftotext >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get install -y poppler-utils >/dev/null
  fi
fi

# ── 3. vLLM 0.14.0 (anchors torch 2.9.1 + cu12 runtime) ─────────────────────
echo "== vLLM 0.14.0 + torch 2.9.1 =="
pip install --quiet "vllm==0.14.0"
python3 -c "
import torch, vllm
print(f'  torch  {torch.__version__}')
print(f'  vllm   {vllm.__version__}')
assert torch.__version__.startswith('2.9'), f'unexpected torch: {torch.__version__}'
"

# ── 4. flash-attn (SM80 Ampere or newer only) ─────────────────────────────────
echo "== flash-attn =="
COMPUTE_MAJOR=$(python3 -c "
import torch, sys
if not torch.cuda.is_available():
    print(0); sys.exit()
print(torch.cuda.get_device_capability(0)[0])
" 2>/dev/null || echo "0")

if [[ "${COMPUTE_MAJOR}" -ge 8 ]]; then
  GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "unknown")
  echo "  GPU: ${GPU_NAME} (SM${COMPUTE_MAJOR}x, Ampere+) — installing wheel"

  # Pre-built wheel: cu12, torch 2.9, cxx11abi=TRUE, cp312, linux x86_64
  FA_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

  if [[ "$PY_MINOR" == "12" ]]; then
    pip install --quiet "$FA_WHEEL" \
      && python3 -c "import flash_attn; print('  flash-attn', flash_attn.__version__, 'OK')" \
      || {
        echo "  Pre-built wheel failed — falling back to source build (takes ~20 min)"
        MAX_JOBS=4 pip install --quiet flash-attn==2.8.3 --no-build-isolation
      }
  else
    echo "  Non-cp312 Python — building from source (takes ~20 min)"
    MAX_JOBS=4 pip install --quiet flash-attn==2.8.3 --no-build-isolation
  fi
else
  echo "  GPU SM${COMPUTE_MAJOR} (pre-Ampere or no GPU) — skipping flash-attn"
  echo "  Training will use attn_implementation=sdpa."
fi

# ── 5. Training deps ──────────────────────────────────────────────────────────
echo "== Training deps (requirements.txt) =="
pip install --quiet -r requirements.txt

# ── 6. Sanity check ───────────────────────────────────────────────────────────
echo "== Sanity =="
python3 -c "
import torch, transformers, accelerate, datasets, peft, vllm, math_verify
print(f'  torch        {torch.__version__}')
print(f'  transformers {transformers.__version__}')
print(f'  accelerate   {accelerate.__version__}')
print(f'  datasets     {datasets.__version__}')
print(f'  peft         {peft.__version__}')
print(f'  vllm         {vllm.__version__}')
print(f'  math_verify  OK')
print(f'  CUDA devices {torch.cuda.device_count()}')

try:
    import flash_attn
    print(f'  flash-attn   {flash_attn.__version__}')
except ImportError:
    print(f'  flash-attn   not installed (SM < 80)')
"

# ── 7. GPU health ─────────────────────────────────────────────────────────────
echo "== GPU health =="
USABLE="$(bash scripts/check_gpus.sh 2>/dev/null || echo '')"
if [[ -z "$USABLE" ]]; then
  echo "WARNING: no usable GPUs detected. Training will fail."
else
  echo "Usable GPU indices: $USABLE"
  echo "  Suggested: export CUDA_VISIBLE_DEVICES=$USABLE"
fi

echo ""
echo "Setup complete."
