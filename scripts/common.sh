#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

DATA_PATH="${DATA_PATH:-${PROJECT_ROOT}/data/nocs}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/runs}"
DETECTOR_CACHE="${DETECTOR_CACHE:-${PROJECT_ROOT}/data/evaluation/segmentation_results_real_test.pkl}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-${PROJECT_ROOT}/checkpoints/stage1.pth}"
STAGE1_AUDIT="${STAGE1_AUDIT:-${PROJECT_ROOT}/checkpoints/stage1_observability_audit.json}"
STAGE2_CHECKPOINT="${STAGE2_CHECKPOINT:-${PROJECT_ROOT}/checkpoints/stage2.pth}"
MASTER_PORT="${MASTER_PORT:-29500}"
PYTHON_BIN="${PYTHON_BIN:-python}"
LEGACY_POINT_OPS_PATH="${LEGACY_POINT_OPS_PATH:-${LEGACY_FPS_PATH:-${PROJECT_ROOT}/third_party/pointnet2}}"

timestamp() {
    date +%Y%m%d_%H%M%S
}

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Missing required command: $1" >&2
        exit 1
    fi
}

require_python() {
    if [[ "${PYTHON_BIN}" == */* ]]; then
        if [[ ! -x "${PYTHON_BIN}" ]]; then
            echo "Python executable does not exist: ${PYTHON_BIN}" >&2
            exit 1
        fi
    else
        require_command "${PYTHON_BIN}"
    fi
}

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "Required file does not exist: $1" >&2
        exit 1
    fi
}

require_directory() {
    if [[ ! -d "$1" ]]; then
        echo "Required directory does not exist: $1" >&2
        exit 1
    fi
}

require_nocs_dataset() {
    require_directory "${DATA_PATH}"
    require_directory "${DATA_PATH}/Real"
    require_directory "${DATA_PATH}/obj_models"
    if [[ "${TRAIN_SOURCE:-CAMERA+Real}" == *CAMERA* ]]; then
        require_directory "${DATA_PATH}/CAMERA"
    fi
}

configure_legacy_point_ops() {
    require_directory "${LEGACY_POINT_OPS_PATH}"
    require_file "${LEGACY_POINT_OPS_PATH}/pointnet2_utils.py"
    if ! compgen -G "${LEGACY_POINT_OPS_PATH}/pointnet2_cuda*.so" >/dev/null; then
        echo "Missing archived CUDA point operators in ${LEGACY_POINT_OPS_PATH}" >&2
        exit 1
    fi
    export INFLUPOSE_FPS_BACKEND=legacy_cuda
    export PYTHONPATH="${LEGACY_POINT_OPS_PATH}:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
}

configure_training_gpus() {
    require_command nvidia-smi
    if [[ -n "${GPU_IDS:-}" ]]; then
        export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
    elif [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' ' | paste -sd, -)"
        export CUDA_VISIBLE_DEVICES
    fi
    if [[ -z "${CUDA_VISIBLE_DEVICES}" ]]; then
        echo "No visible CUDA devices were found." >&2
        exit 1
    fi
    local visible_without_commas="${CUDA_VISIBLE_DEVICES//,/}"
    GPU_COUNT="${GPU_COUNT:-$(( ${#CUDA_VISIBLE_DEVICES} - ${#visible_without_commas} + 1 ))}"
    if (( GPU_COUNT < 1 )); then
        echo "GPU_COUNT must be positive." >&2
        exit 1
    fi
    export GPU_COUNT
}

configure_eval_device() {
    DEVICE="${DEVICE:-cuda:0}"
    if [[ "${DEVICE}" == cuda:* ]]; then
        require_command nvidia-smi
        export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
        DEVICE="cuda:0"
    fi
    export DEVICE
}

launch_training() {
    local module="$1"
    shift
    cd "${PROJECT_ROOT}"
    require_python
    if (( GPU_COUNT > 1 )); then
        "${PYTHON_BIN}" -m torch.distributed.run \
            --standalone \
            --nproc_per_node="${GPU_COUNT}" \
            --master_port="${MASTER_PORT}" \
            -m "${module}" \
            --distributed \
            "$@"
    else
        "${PYTHON_BIN}" -m "${module}" --device cuda:0 "$@"
    fi
}
