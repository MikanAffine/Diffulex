#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_REAL_MODEL_TEST=0
exec "$SCRIPT_DIR/test_multi_bd_prefix_caching.sh"
