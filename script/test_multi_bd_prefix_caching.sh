#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="${PYTHON_BIN:-python}"
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export CUDA_VISIBLE_DEVICES

MODEL_PATH="${MODEL_PATH:-/data1/ckpts/JetLM/SDAR-1.7B-Chat-b32}"
MODEL_NAME="${MODEL_NAME:-sdar}"
MASTER_PORT="${MASTER_PORT:-$((2440 + ($$ % 200)))}"
RUN_REAL_MODEL_TEST="${RUN_REAL_MODEL_TEST:-1}"

echo "== Multi-BD Prefix Caching Validation =="
echo "project_root=$PROJECT_ROOT"
echo "python=$PYTHON_BIN"
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
echo "master_port=$MASTER_PORT"
echo "run_real_model_test=$RUN_REAL_MODEL_TEST"

echo
echo "[1/3] Engine regressions"
"$PYTHON_BIN" -m pytest -q \
  test/python/engine/test_cached_prefix_prefill.py \
  test/python/engine/test_strategy_runtime_config.py \
  test/python/engine/test_multi_bd_kv_cache_manager.py \
  test/python/engine/test_monkey_patch.py

echo
echo "[2/3] Kernel regressions"
"$PYTHON_BIN" -m pytest -q \
  test/python/kernel/test_dllm_flash_attn_chunked_prefill_unified_kernel.py \
  -k 'prefix_prefill_varlen_block_causal or chunked_prefill_mixed_all_have_cache or prefix_full_with_cache'

if [ "$RUN_REAL_MODEL_TEST" != "1" ]; then
    echo
    echo "[3/3] Real model comparison skipped"
    exit 0
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "MODEL_PATH does not exist: $MODEL_PATH" >&2
    exit 1
fi

echo
echo "[3/3] Real model comparison"
"$PYTHON_BIN" script/debug_compare_prefix_caching.py \
  --model "$MODEL_PATH" \
  --model-name "$MODEL_NAME" \
  --master-port "$MASTER_PORT"

echo
echo "All prefix-caching validations passed."
