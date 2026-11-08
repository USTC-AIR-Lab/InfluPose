#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

CHECKPOINT="${CHECKPOINT:-${STAGE1_CHECKPOINT}}"
OUTPUT="${OUTPUT:-${RUN_ROOT}/stage1_observability_audit.json}"
EXAMPLES="${EXAMPLES:-32}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"

require_nocs_dataset
require_file "${CHECKPOINT}"
configure_legacy_point_ops
configure_eval_device
require_python
mkdir -p "$(dirname -- "${OUTPUT}")"

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" -m stage1.audit_observability \
    --data_path "${DATA_PATH}" \
    --checkpoint "${CHECKPOINT}" \
    --output "${OUTPUT}" \
    --examples "${EXAMPLES}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --point_ops_backend legacy_cuda \
    --device "${DEVICE}" \
    "$@"

echo "Stage1 observability audit: ${OUTPUT}"
