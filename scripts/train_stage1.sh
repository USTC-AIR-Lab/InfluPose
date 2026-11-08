#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

TRAIN_SOURCE="${TRAIN_SOURCE:-CAMERA+Real}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_ROOT}/stage1_full_$(timestamp)}"
EPOCHS="${EPOCHS:-36}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DISABLE_PERSISTENT_WORKERS="${DISABLE_PERSISTENT_WORKERS:-0}"
CAMERA_REAL_RATIO="${CAMERA_REAL_RATIO:-3.0}"
SEED="${SEED:-260892}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
LINEAGE_MANIFEST="${LINEAGE_MANIFEST:-}"
GPU_IDS="${GPU_IDS:-0}"
export GPU_IDS

require_nocs_dataset
configure_legacy_point_ops
configure_training_gpus
mkdir -p "${OUTPUT_DIR}"

arguments=(
    --data_path "${DATA_PATH}"
    --output_dir "${OUTPUT_DIR}"
    --train_source "${TRAIN_SOURCE}"
    --epochs "${EPOCHS}"
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --num_points 1024
    --camera_real_ratio "${CAMERA_REAL_RATIO}"
    --point_ops_backend legacy_cuda
    --max_train_batches -1
    --seed "${SEED}"
)
if [[ "${DISABLE_PERSISTENT_WORKERS}" == "1" ]]; then
    arguments+=(--disable_persistent_workers)
elif [[ "${DISABLE_PERSISTENT_WORKERS}" != "0" ]]; then
    echo "DISABLE_PERSISTENT_WORKERS must be 0 or 1." >&2
    exit 1
fi
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
    require_file "${RESUME_CHECKPOINT}"
    arguments+=(--resume "${RESUME_CHECKPOINT}")
fi
if [[ -n "${LINEAGE_MANIFEST}" ]]; then
    require_file "${LINEAGE_MANIFEST}"
    arguments+=(--lineage_manifest "${LINEAGE_MANIFEST}")
fi

echo "Stage1 full training"
echo "  data: ${DATA_PATH}"
echo "  output: ${OUTPUT_DIR}"
echo "  GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "  global batch: $(( BATCH_SIZE * GPU_COUNT ))"
launch_training stage1.train "${arguments[@]}" "$@"
