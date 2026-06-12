#!/usr/bin/env bash
# serve_gemma4-31b-fp8-tpN.sh
#
# Bare-metal vLLM serve for Gemma 4 31B DENSE FP8 on Intel Arc Pro B70 (BMG-G31),
# parameterized over tensor-parallel size, card mask, context window, and gpu-util.
# Single-card (TP=1) and multi-card (TP=2 / TP=4) both supported. Applies the
# triton init_devices patch if missing. See docs/01-baremetal-vllm.md and
# docs/02-tensor-parallel.md.
#
# Env overrides:
#   VENV_DIR        path to the Python 3.12 venv with the coupled +xpu wheel set
#   ONEAPI_VARS     path to oneAPI 2025.3 oneapi-vars.sh (vLLM's torch+xpu was built against 2025.3)
#   TP_SIZE         tensor-parallel size           default: 4
#   MAX_MODEL_LEN   per-query context window        default: 16384
#   GPU_MEM_UTIL    per-card mem fraction           default: 0.80   (TP=4 0.90 CRASHES; TP=2 use 0.90)
#   ZE_AFFINITY_MASK card mask                      default: first TP_SIZE cards (0,1,...)
#   MODEL           weights                          default: RedHatAI/gemma-4-31B-it-FP8-Dynamic
#   PORT            host port                        default: 8003
#
# Endpoint: http://127.0.0.1:${PORT}/v1   Served name: gemma-4-31B-it
set -eo pipefail

VENV_DIR="${VENV_DIR:?set VENV_DIR to your vLLM +xpu venv}"
ONEAPI_VARS="${ONEAPI_VARS:-/opt/intel/oneapi/2025.3/oneapi-vars.sh}"
TP_SIZE="${TP_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.80}"
PORT="${PORT:-8003}"
MODEL="${MODEL:-RedHatAI/gemma-4-31B-it-FP8-Dynamic}"

# Default card mask = first TP_SIZE cards (0,1,...,TP_SIZE-1) unless overridden.
if [ -z "${ZE_AFFINITY_MASK:-}" ]; then
  mask="0"; for i in $(seq 1 $((TP_SIZE-1))); do mask="$mask,$i"; done
  ZE_AFFINITY_MASK="$mask"
fi

# vLLM's torch+xpu was built against oneAPI 2025.3 -> source it with --force.
[ -f "$ONEAPI_VARS" ] || { echo "FATAL: oneAPI 2025.3 missing at $ONEAPI_VARS"; exit 2; }
set +u; source "$ONEAPI_VARS" --force >/dev/null; set -u
command -v icx >/dev/null || { echo "FATAL: icx not on PATH after sourcing oneAPI 2025.3"; exit 3; }

# --- multi-root + oneCCL env (validated for B70 TP) --------------------------
export ZE_AFFINITY_MASK
export ONEAPI_DEVICE_SELECTOR='level_zero:*'
export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
export SYCL_UR_USE_LEVEL_ZERO_V2=0
export CCL_ENABLE_SYCL_KERNELS=0
export CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0
export CCL_ATL_TRANSPORT=ofi
export CCL_ZE_IPC_EXCHANGE=pidfd
export FI_PROVIDER=shm
export CCL_ATL_SHM=1
export CCL_WORKER_COUNT="$TP_SIZE"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
export ZES_ENABLE_SYSMAN=1
export TRITON_INTEL_DEVICE_ARCH=bmg     # mandatory for TP (arch auto-parse fails under spawn workers)
unset SYCL_CACHE_PERSISTENT || true     # NEVER persist the SYCL cache on BMG

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
python -c "import vllm" 2>/dev/null || { echo "FATAL: vllm not importable in $VENV_DIR"; exit 6; }

# triton init_devices patch must be present (any triton kernel triggers the bug)
DRV="$VENV_DIR/lib/python3.12/site-packages/triton/backends/intel/driver.c"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$DRV" ] && ! grep -q 'PATCHED: tolerate missing OpenCL' "$DRV" 2>/dev/null; then
  echo "[serve] applying triton init_devices patch + clearing ~/.triton/cache"
  python "$SCRIPT_DIR/patch_triton_driver_noopenclsel.py" "$DRV" || true
  rm -rf "$HOME/.triton/cache"
fi

export HF_HUB_OFFLINE=1
echo "[serve] model=$MODEL TP=$TP_SIZE mask=$ZE_AFFINITY_MASK win=$MAX_MODEL_LEN util=$GPU_MEM_UTIL port=$PORT"
echo "[serve] vllm=$(python -c 'import vllm; print(vllm.__version__)')"

exec vllm serve "$MODEL" \
  --tensor-parallel-size "$TP_SIZE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --kv-cache-dtype fp8 \
  --enforce-eager \
  --attention-backend TRITON_ATTN \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --reasoning-parser gemma4 \
  --served-model-name gemma-4-31B-it \
  --host 127.0.0.1 \
  --port "$PORT"
