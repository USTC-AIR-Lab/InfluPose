#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

CHECKPOINT="${CHECKPOINT:-${STAGE1_CHECKPOINT}}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_ROOT}/stage1_real275_$(timestamp)}"
BATCH_SIZE="${BATCH_SIZE:-128}"

require_nocs_dataset
require_file "${CHECKPOINT}"
require_file "${DETECTOR_CACHE}"
configure_legacy_point_ops
configure_eval_device
require_python
mkdir -p "${OUTPUT_DIR}"

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" -m stage1.evaluate_real275 \
    --data_path "${DATA_PATH}" \
    --checkpoint "${CHECKPOINT}" \
    --detector_cache "${DETECTOR_CACHE}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --point_ops_backend legacy_cuda \
    --device "${DEVICE}" \
    "$@"
