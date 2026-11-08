#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

CHECKPOINT="${CHECKPOINT:-${STAGE2_CHECKPOINT}}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_ROOT}/stage2_real275_$(timestamp)}"
BATCH_SIZE="${BATCH_SIZE:-48}"
MAX_INSTANCES="${MAX_INSTANCES:-0}"
WEIGHTS="${WEIGHTS:-auto}"
FPS_BACKEND="${FPS_BACKEND:-auto}"
require_file "${CHECKPOINT}"
require_file "${DETECTOR_CACHE}"
configure_eval_device
require_python
mkdir -p "${OUTPUT_DIR}"
if [[ "${FPS_BACKEND}" != "torch" ]]; then
    configure_legacy_point_ops
fi

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" -m stage2.evaluate_real275 \
    --checkpoint "${CHECKPOINT}" \
    --data_path "${DATA_PATH}" \
    --detector_cache "${DETECTOR_CACHE}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --max_instances "${MAX_INSTANCES}" \
    --weights "${WEIGHTS}" \
    --fps_backend "${FPS_BACKEND}" \
    --evaluation_irls_iterations "${IRLS_ITERATIONS:-1}" \
    --device "${DEVICE}" \
    "$@"
