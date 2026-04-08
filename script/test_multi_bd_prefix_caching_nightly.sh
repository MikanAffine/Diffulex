#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "${MODEL_PATH:-}" ]; then
  export MODEL_PATH="/data1/ckpts/JetLM/SDAR-1.7B-Chat-b32"
fi

if [ ! -d "${MODEL_PATH}" ]; then
  echo "MODEL_PATH does not exist: ${MODEL_PATH}" >&2
  exit 1
fi

export RUN_REAL_MODEL_TEST=1
exec "$SCRIPT_DIR/test_multi_bd_prefix_caching.sh"
