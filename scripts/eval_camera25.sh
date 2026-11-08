#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
require_nocs_dataset
require_file "${STAGE2_CHECKPOINT}"
configure_legacy_point_ops
configure_eval_device
require_python
cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" -m stage2.evaluate_camera25 \
    --data_path "${DATA_PATH}" \
    --checkpoint "${STAGE2_CHECKPOINT}" \
    --output_dir "${OUTPUT_DIR:-runs/camera25}" \
    --batch_size "${BATCH_SIZE:-48}" \
    --num_workers "${NUM_WORKERS:-4}" \
    --weights ema --irls_iterations "${IRLS_ITERATIONS:-1}" \
    --device "${DEVICE}" "$@"
