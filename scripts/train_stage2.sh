#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# A local dataset file is optional.  Once it has been created from the
# repository template, the canonical Stage2 command becomes simply
# `bash scripts/train_stage2.sh`.  Precedence is CLI > process environment >
# local config > built-in defaults.
STAGE2_CONFIG_EXPLICIT=0
STAGE2_CONFIG_FILE="${STAGE2_CONFIG:-${PROJECT_ROOT}/configs/stage2_dataset.env}"
stage2_argv=("$@")
for ((stage2_arg_index = 0; stage2_arg_index < ${#stage2_argv[@]}; stage2_arg_index++)); do
    if [[ "${stage2_argv[stage2_arg_index]}" == "--config" ]]; then
        if (( stage2_arg_index + 1 >= ${#stage2_argv[@]} )); then
            echo "--config requires a value" >&2
            exit 2
        fi
        STAGE2_CONFIG_FILE="${stage2_argv[stage2_arg_index + 1]}"
        STAGE2_CONFIG_EXPLICIT=1
        break
    fi
done

# Preserve values explicitly supplied by the caller before sourcing the local
# site config.  This makes `DATA_PATH=/new/data bash scripts/train_stage2.sh`
# a true one-command override while still allowing the config to provide
# workstation defaults.  CLI options are parsed after this block and therefore
# remain the highest-precedence source.
stage2_environment_keys=(
    DATASET_ADAPTER
    DATA_PATH
    OUTPUT_DIR
    GPU_IDS
    BATCH_SIZE
    NUM_WORKERS
    MASTER_PORT
    PYTHON_BIN
    STAGE1_CHECKPOINT
    STAGE1_AUDIT
    LEGACY_POINT_OPS_PATH
    TRAIN_SOURCE
    TRAIN_SEED
)
declare -A stage2_environment_was_set=()
declare -A stage2_environment_value=()
for stage2_environment_key in "${stage2_environment_keys[@]}"; do
    if [[ -v "${stage2_environment_key}" ]]; then
        stage2_environment_was_set["${stage2_environment_key}"]=1
        stage2_environment_value["${stage2_environment_key}"]="${!stage2_environment_key}"
    fi
done
if [[ -f "${STAGE2_CONFIG_FILE}" ]]; then
    # shellcheck source=/dev/null
    source "${STAGE2_CONFIG_FILE}"
elif (( STAGE2_CONFIG_EXPLICIT )); then
    echo "Stage2 config does not exist: ${STAGE2_CONFIG_FILE}" >&2
    exit 2
fi
for stage2_environment_key in "${stage2_environment_keys[@]}"; do
    if [[ -n "${stage2_environment_was_set[${stage2_environment_key}]:-}" ]]; then
        printf -v "${stage2_environment_key}" '%s' \
            "${stage2_environment_value[${stage2_environment_key}]}"
    fi
done

: "${PYTHON_BIN:=python}"
export PYTHON_BIN
STAGE2_MASTER_PORT_OVERRIDE="${MASTER_PORT:-}"
source "${SCRIPT_DIR}/common.sh"

usage() {
    cat <<'EOF'
Usage: bash scripts/train_stage2.sh [options] [-- stage2.train args]

Default Stage2 training is one reproducible 50-epoch recipe:

  Start from random Stage2 weights (seed 260907).
  Train 30 base epochs, freeze their EMA reference, then train 20 refinement
  epochs. The phase transition and checkpoint continuation are automatic.

Six GPUs, batch 6 per GPU, global batch 36. Interrupted runs resume
from latest.pth. Training uses three IRLS updates; evaluation uses one.

New processed dataset example:

  bash scripts/train_stage2.sh \
    --data-path /absolute/path/to/data \
    --stage1-checkpoint /absolute/path/to/stage1/best.pth \
    --stage1-audit /absolute/path/to/stage1/observability_audit.json

One-command setup for repeated runs:

  cp configs/stage2_dataset.env.example configs/stage2_dataset.env
  # Edit the dataset/teacher paths once, then run:
  bash scripts/train_stage2.sh

Options:
  --config FILE           Load dataset/site paths from a shell env file.
  --dataset NAME           Dataset adapter: nocs.
  --data-path PATH          Processed dataset root.
  --output-dir PATH         Training output root; otherwise derived safely.
  --stage1-checkpoint PATH  Frozen Stage1 teacher checkpoint.
  --stage1-audit PATH       Audit JSON bound to the Stage1 checkpoint.
  --gpu-ids IDS             Six comma-separated GPU indices.
  --print-config            Print the machine-readable default protocol.
  --dry-run                 Validate inputs and print the training launch.

Precedence: CLI options > process environment > config file > built-in defaults.

Useful overrides: DATA_PATH, OUTPUT_DIR, GPU_IDS, BATCH_SIZE, NUM_WORKERS,
MASTER_PORT, PYTHON_BIN, STAGE1_CHECKPOINT, STAGE1_AUDIT, LEGACY_POINT_OPS_PATH.
EOF
}

DRY_RUN=0
PRINT_CONFIG=0
DATASET_ADAPTER="${DATASET_ADAPTER:-nocs}"
EXTRA_ARGUMENTS=()
while (($#)); do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --print-config)
            PRINT_CONFIG=1
            ;;
        --dry-run)
            DRY_RUN=1
            ;;
        --config)
            if (($# < 2)); then
                echo "$1 requires a value" >&2
                exit 2
            fi
            STAGE2_CONFIG_FILE="$2"
            shift
            ;;
        --dataset|--data-path|--output-dir|--stage1-checkpoint|--stage1-audit|--gpu-ids)
            if (($# < 2)); then
                echo "$1 requires a value" >&2
                exit 2
            fi
            option="$1"
            value="$2"
            case "${option}" in
                --dataset) DATASET_ADAPTER="${value}" ;;
                --data-path) DATA_PATH="${value}" ;;
                --output-dir) OUTPUT_DIR="${value}" ;;
                --stage1-checkpoint) STAGE1_CHECKPOINT="${value}" ;;
                --stage1-audit) STAGE1_AUDIT="${value}" ;;
                --gpu-ids) GPU_IDS="${value}" ;;
            esac
            shift
            ;;
        --)
            shift
            EXTRA_ARGUMENTS+=("$@")
            break
            ;;
        *)
            EXTRA_ARGUMENTS+=("$1")
            ;;
    esac
    shift
done

case "${DATASET_ADAPTER}" in
    nocs)
        TRAIN_MODULE="stage2.train"
        TRAIN_SOURCE="${TRAIN_SOURCE:-CAMERA+Real}"
        ;;
    *)
        echo "--dataset must be nocs; resolved ${DATASET_ADAPTER}" >&2
        exit 2
        ;;
esac
TRAIN_SEED="${TRAIN_SEED:-260907}"
if [[ -z "${OUTPUT_DIR:-}" ]]; then
    resolved_data_path="$(realpath -m -- "${DATA_PATH}")"
    canonical_default_data_path="$(realpath -m -- "${PROJECT_ROOT}/data/nocs")"
    if [[ "${resolved_data_path}" == "${canonical_default_data_path}" ]]; then
        OUTPUT_DIR="${PROJECT_ROOT}/results/stage2/full50_seed${TRAIN_SEED}"
    else
        dataset_leaf="$(basename -- "${resolved_data_path}")"
        if [[ "${dataset_leaf}" == "data" ]]; then
            dataset_leaf="$(basename -- "$(dirname -- "${resolved_data_path}")")"
        fi
        dataset_tag="$(printf '%s' "${dataset_leaf}" | tr -cs '[:alnum:]_.-' '_')"
        dataset_hash="$(printf '%s' "${resolved_data_path}" | sha256sum | cut -c1-8)"
        OUTPUT_DIR="${PROJECT_ROOT}/results/stage2/full50_${dataset_tag}_${dataset_hash}_seed${TRAIN_SEED}"
    fi
fi
FULL50_DIR="${FULL50_DIR:-${OUTPUT_DIR}/full50}"
FULL50_CHECKPOINT="${FULL50_CHECKPOINT:-${FULL50_DIR}/epoch_050.pth}"
PROTOCOL_MANIFEST="${PROTOCOL_MANIFEST:-${PROJECT_ROOT}/stage2/default_training_protocol.json}"
RUN_CONTRACT="${RUN_CONTRACT:-${OUTPUT_DIR}/run_contract.json}"
RUN_PROTOCOL="${RUN_PROTOCOL:-${OUTPUT_DIR}/default_training_protocol.json}"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5}"
BATCH_SIZE="${BATCH_SIZE:-6}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MASTER_PORT="${STAGE2_MASTER_PORT_OVERRIDE:-29641}"
export GPU_IDS MASTER_PORT

if (( PRINT_CONFIG )); then
    require_file "${PROTOCOL_MANIFEST}"
    sed -n '1,260p' "${PROTOCOL_MANIFEST}"
    exit 0
fi

require_python
require_nocs_dataset
require_file "${STAGE1_CHECKPOINT}"
require_file "${STAGE1_AUDIT}"
require_file "${PROTOCOL_MANIFEST}"
configure_legacy_point_ops
configure_training_gpus

if (( GPU_COUNT != 6 )); then
    echo "Default Stage2 requires exactly 6 visible GPUs; resolved ${GPU_COUNT}." >&2
    echo "Set GPU_IDS to six device indices, for example 0,1,2,3,4,5." >&2
    exit 1
fi
if (( BATCH_SIZE != 6 )); then
    echo "Default Stage2 requires per-GPU batch 6; resolved ${BATCH_SIZE}." >&2
    exit 1
fi
GLOBAL_BATCH="$(( BATCH_SIZE * GPU_COUNT ))"
if (( GLOBAL_BATCH != 36 )); then
    echo "Default Stage2 requires global batch 36; resolved ${GLOBAL_BATCH}." >&2
    exit 1
fi

ensure_run_contract() {
    if (( DRY_RUN )); then
        echo "  run contract: ${RUN_CONTRACT} (validated when training starts)"
        return
    fi
    mkdir -p "${OUTPUT_DIR}"
    "${PYTHON_BIN}" - \
        "${RUN_CONTRACT}" \
        "${DATA_PATH}" \
        "${TRAIN_SOURCE}" \
        "${DATASET_ADAPTER}" \
        "${STAGE1_CHECKPOINT}" \
        "${STAGE1_AUDIT}" \
        "${GPU_COUNT}" \
        "${BATCH_SIZE}" \
        "${TRAIN_SEED}" \
        "${PROTOCOL_MANIFEST}" \
        "$([[ -f "${STAGE2_CONFIG_FILE}" ]] && realpath -m -- "${STAGE2_CONFIG_FILE}" || true)" \
        "${NUM_WORKERS}" "${EXTRA_ARGUMENTS[@]}" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys


def digest(path: str) -> str:
    value = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


(
    contract_path,
    data_path,
    train_source,
    dataset_adapter,
    stage1_checkpoint,
    stage1_audit,
    gpu_count,
    batch_size,
    train_seed,
    protocol_manifest,
    local_config,
    num_workers,
    *extra_arguments,
) = sys.argv[1:]
desired = {
    "format": "STAGE2_BASE30_REFINE20_RUN_CONTRACT_V3",
    "num_workers": int(num_workers),
    "extra_arguments": extra_arguments,
    "data_path": os.path.realpath(os.path.abspath(data_path)),
    "train_source": train_source,
    "stage1_checkpoint": os.path.realpath(os.path.abspath(stage1_checkpoint)),
    "stage1_checkpoint_sha256": digest(stage1_checkpoint),
    "stage1_audit": os.path.realpath(os.path.abspath(stage1_audit)),
    "stage1_audit_sha256": digest(stage1_audit),
    "protocol": "random_init_base30_refine20_v6",
    "gpu_count": int(gpu_count),
    "per_gpu_batch": int(batch_size),
    "global_batch": int(gpu_count) * int(batch_size),
    "train_seed": int(train_seed),
    "protocol_manifest": os.path.realpath(os.path.abspath(protocol_manifest)),
    "protocol_manifest_sha256": digest(protocol_manifest),
}
if local_config:
    desired["local_config"] = os.path.realpath(os.path.abspath(local_config))
    desired["local_config_sha256"] = digest(local_config)
if dataset_adapter != "nocs":
    desired["dataset_adapter"] = dataset_adapter
target = Path(contract_path)
if target.exists():
    current = json.loads(target.read_text(encoding="utf-8"))
    if current != desired:
        differences = {
            key: {"recorded": current.get(key), "requested": desired.get(key)}
            for key in sorted(set(current) | set(desired))
            if current.get(key) != desired.get(key)
        }
        raise SystemExit(
            "Stage2 output directory is bound to another run contract. "
            "Set OUTPUT_DIR to a new directory. Differences: "
            + json.dumps(differences, sort_keys=True)
        )
else:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(desired, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, target)
PY
    if [[ ! -f "${RUN_PROTOCOL}" ]] || ! cmp -s "${PROTOCOL_MANIFEST}" "${RUN_PROTOCOL}"; then
        cp -- "${PROTOCOL_MANIFEST}" "${RUN_PROTOCOL}"
    fi
}

print_command() {
    printf 'DRY-RUN:'
    printf ' %q' "$@"
    printf '\n'
}

run_training() {
    local log_path="$1"
    shift
    if (( DRY_RUN )); then
        print_command "${PYTHON_BIN}" -m torch.distributed.run \
            --standalone --nproc_per_node=6 --master_port="${MASTER_PORT}" \
            -m "${TRAIN_MODULE}" --distributed "$@"
        return
    fi
    mkdir -p "$(dirname -- "${log_path}")"
    launch_training "${TRAIN_MODULE}" "$@" 2>&1 | tee -a "${log_path}"
}

architecture_arguments=(
    --num_points 1024
    --normals_k 16
    --fps_backend legacy_cuda
    --token_width 192
    --anchor_count 192
    --relation_layers 4
    --heads 6
    --dropout 0.05
    --irls_iterations 3
    --irls_damping 1e-3
    --max_rotation_step_degrees 8.0
    --max_translation_step 0.015
    --max_log_size_step 0.12
)

teacher_arguments=(
    --stage1_checkpoint "${STAGE1_CHECKPOINT}"
    --observability_audit "${STAGE1_AUDIT}"
    --observability_interval 2
    --observability_fraction 0.5
    --observability_regions 8
    --observability_removal_fraction 0.125
    --observability_teacher_chunk 24
    --teacher_label_blend 0.70
)

training_arguments=(
    --data_path "${DATA_PATH}"
    --train_source "${TRAIN_SOURCE}"
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --camera_real_ratio 1.0
    --learning_rate 3e-4
    --head_learning_rate 7e-4
    --weight_decay 1e-4
    --ema_decay 0.999
    --grad_clip 5.0
    --amp bfloat16
    --view_consistency_weight 0.5
    --view_fraction 0.25
    --view_interval 4
    --view_dropout_max 0.25
    --view_noise_std 0.0025
    --view_axial_noise_std 0.004
    --reference_pose_weight 2.0
    --reference_iou_weight 2.0
    --reference_rotation_margin_degrees 0.25
    --reference_translation_margin_cm 0.10
    --reference_iou_margin 0.01
    "${architecture_arguments[@]}"
    "${teacher_arguments[@]}"
)

echo "Stage2 default full50 protocol"
echo "  dataset adapter: ${DATASET_ADAPTER} (${TRAIN_MODULE})"
echo "  data: ${DATA_PATH}"
echo "  output: ${OUTPUT_DIR}"
if [[ -f "${STAGE2_CONFIG_FILE}" ]]; then
    echo "  local config: ${STAGE2_CONFIG_FILE}"
fi
echo "  GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "  per-GPU/global batch: ${BATCH_SIZE}/${GLOBAL_BATCH}"
echo "  training: random initialization -> 30 base epochs -> 20 refinement epochs"
ensure_run_contract

if [[ ! -f "${FULL50_CHECKPOINT}" ]]; then
    full50_arguments=(
        "${training_arguments[@]}"
        --output_dir "${FULL50_DIR}"
        --epochs 50
        --base_epochs 30
        --refine_learning_rate 1.5e-5
        --refine_head_learning_rate 3e-5
        --refine_eta_min 3e-7
        --refine_camera_real_ratio 0.75
        --max_train_batches -1
        --seed "${TRAIN_SEED}"
    )
    if [[ -f "${FULL50_DIR}/latest.pth" ]]; then
        full50_arguments+=(--resume "${FULL50_DIR}/latest.pth")
        echo "  full50: resuming exact optimizer/scheduler/EMA state"
    else
        echo "  full50: starting 30 + 20 epochs from random weights"
    fi
    run_training "${FULL50_DIR}/train.log" \
        "${full50_arguments[@]}" "${EXTRA_ARGUMENTS[@]}"
else
    echo "  full50: complete (${FULL50_CHECKPOINT})"
fi

if (( ! DRY_RUN )); then
    require_file "$FULL50_CHECKPOINT"
    ln -sfn "$(realpath -- "$FULL50_CHECKPOINT")" "$OUTPUT_DIR/final.pth"
fi
if (( DRY_RUN )); then
    echo "Stage2 full50 dry-run complete (no training was started)"
else
    echo "Stage2 full50 training complete"
fi
echo "  resumable checkpoint: $FULL50_CHECKPOINT"
echo "  stable final link: $OUTPUT_DIR/final.pth"
