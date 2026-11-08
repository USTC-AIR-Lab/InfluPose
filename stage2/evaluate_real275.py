"""Evaluate the deterministic Stage2 information-field endpoint on REAL275."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle

import numpy as np
import torch
from tqdm import tqdm

from shared import point_ops
from shared.point_features import estimate_camera_observation_features
from shared.pose_geometry import continuous_y_symmetry_mask, pose6d_to_matrix
from shared.real275_io import empty_prediction, records_from_detection_cache
from shared.real275_metrics import (
    compare_per_category_reference,
    compute_real275_3d_map,
    per_class_diagnostics,
    summary_from_metrics,
)
from stage2.model import Stage2PoseInformationField

CHECKPOINT_FORMAT = "STAGE2_G10F_INFORMATION_FIELD_IRLS_V1"
PRE_RELEASE_G10F_CHECKPOINT_FORMAT = "STAGE2_POSE_INFORMATION_FIELD_V1"
LEGACY_CHECKPOINT_FORMAT = "STAGE2_POSE_INFORMATION_FIELD_V2"
ARCHIVED_G10F_CHECKPOINT_FORMAT = (
    "G10F_OBSERVABILITY_GUIDED_POSE_INFORMATION_FIELD_V1"
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/stage2.pth")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--data_path", default=os.environ.get("DATA_PATH", "data/nocs"))
    parser.add_argument(
        "--detector_cache",
        default="data/evaluation/segmentation_results_real_test.pkl",
    )
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--normals_k", type=int, default=16)
    parser.add_argument("--max_instances", type=int, default=0)
    parser.add_argument("--weights", choices=("auto", "ema", "raw"), default="auto")
    parser.add_argument(
        "--fps_backend",
        choices=("auto", point_ops.TORCH_FPS_BACKEND, point_ops.LEGACY_FPS_BACKEND),
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=260925)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--evaluation_irls_iterations",
        type=int,
        default=1,
        help=(
            "Number of robust solver updates at inference (default: 1)."
        ),
    )
    return parser.parse_args()


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_epoch_number(state: dict) -> int:
    """Return the one-based epoch number recorded by a checkpoint."""
    index_base = state.get("epoch_index_base", 0)
    if index_base not in (0, 1):
        raise ValueError("checkpoint epoch_index_base must be 0 or 1")
    if "epoch" not in state:
        return 0
    return int(state["epoch"]) + 1 - index_base


def load_model(
    path: str, device: str, weights: str
) -> tuple[Stage2PoseInformationField, dict]:
    state = torch.load(path, map_location="cpu")
    checkpoint_format = state.get("format")
    if checkpoint_format not in {
        CHECKPOINT_FORMAT,
        PRE_RELEASE_G10F_CHECKPOINT_FORMAT,
        LEGACY_CHECKPOINT_FORMAT,
        ARCHIVED_G10F_CHECKPOINT_FORMAT,
    }:
        raise ValueError("checkpoint is not Stage2")
    if state.get("class_prior_audit", {}).get("passed") is not True:
        raise ValueError("Stage2 checkpoint lacks a passing class-prior audit")
    if state.get("contract_audit", {}).get("passed") is not True:
        raise ValueError("Stage2 checkpoint lacks a passing endpoint audit")
    provenance = state.get("provenance", {})
    required_false = (
        (
            "uses_rgb",
            "uses_completion",
            "uses_latent_prediction",
            "uses_g10a_at_inference",
            "uses_category_network_input",
            "uses_genpose_weights",
        )
        if checkpoint_format == ARCHIVED_G10F_CHECKPOINT_FORMAT
        else (
            "uses_rgb",
            "external_pretrained_weights",
            "generative_completion",
            "latent_prediction",
            "stage1_at_inference",
            "category_network_input",
        )
    )
    if any(provenance.get(key, True) for key in required_false):
        raise ValueError("Stage2 checkpoint violates the declared inference route")
    if provenance.get("single_deterministic_pose") is not True:
        raise ValueError("Stage2 checkpoint does not document a single endpoint")
    if weights == "auto":
        weights = str(state.get("recommended_evaluation_weights", "ema"))
    if weights not in {"ema", "raw"}:
        raise ValueError("recommended checkpoint weights must be 'ema' or 'raw'")
    model = Stage2PoseInformationField.from_architecture(state["architecture"])
    selected = (
        state.get("ema_model_state_dict", state["model_state_dict"])
        if weights == "ema"
        else state["model_state_dict"]
    )
    model.load_state_dict(selected, strict=True)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    state = dict(state)
    state["resolved_evaluation_weights"] = weights
    state["restored_solver_contract"] = (
        "three_iteration_anisotropic_information_weighted_irls"
    )
    return model, state


def write_batch(
    prediction: dict,
    records: dict,
    start: int,
    pose: torch.Tensor,
    size: torch.Tensor,
) -> None:
    rotations = pose6d_to_matrix(pose[:, :6]).cpu().numpy()
    translations = pose[:, 6:].cpu().numpy()
    sizes = size.cpu().numpy()
    for offset in range(len(rotations)):
        image = records["image"][start + offset]
        prediction_index = records["prediction_index"][start + offset]
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotations[offset]
        transform[:3, 3] = translations[offset]
        prediction[image]["result"]["pred_RTs"][prediction_index] = transform
        prediction[image]["result"]["pred_scales"][prediction_index] = sizes[offset]


def main() -> None:
    cfg = arguments()
    if cfg.batch_size < 1:
        raise ValueError("batch size must be positive")
    if cfg.max_instances < 0 or cfg.evaluation_irls_iterations < -1:
        raise ValueError("max_instances or evaluation solver depth is invalid")
    if cfg.max_instances:
        raise ValueError("REAL275 evaluation requires complete frames; use max_instances=0")
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if cfg.device.startswith("cuda"):
        torch.cuda.manual_seed_all(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)
    model, state = load_model(cfg.checkpoint, cfg.device, cfg.weights)
    training_irls_iterations = int(model.irls_iterations)
    evaluation_irls_iterations = (
        training_irls_iterations
        if cfg.evaluation_irls_iterations < 0
        else int(cfg.evaluation_irls_iterations)
    )
    model.irls_iterations = evaluation_irls_iterations
    restored_solver_contract = (
        "inlier_precision_influence_weighted_kabsch_only"
        if evaluation_irls_iterations == 0
        else (
            f"{evaluation_irls_iterations}_iteration_"
            "anisotropic_information_weighted_irls"
        )
    )
    state["restored_solver_contract"] = restored_solver_contract
    checkpoint_backend = str(
        state.get("training_config", {}).get(
            "fps_backend",
            state.get("provenance", {}).get(
                "fps_backend", point_ops.LEGACY_FPS_BACKEND
            ),
        )
    )
    resolved_fps_backend = (
        checkpoint_backend if cfg.fps_backend == "auto" else cfg.fps_backend
    )
    os.environ["INFLUPOSE_FPS_BACKEND"] = resolved_fps_backend
    with open(cfg.detector_cache, "rb") as stream:
        detector_cache = pickle.load(stream)
    expected_points = int(state.get("training_config", {}).get("num_points", 1024))
    records, input_stats = records_from_detection_cache(detector_cache, expected_points)
    if cfg.max_instances and len(records["image"]) > cfg.max_instances:
        for key in records:
            records[key] = records[key][: cfg.max_instances]
        input_stats["preview_max_instances"] = cfg.max_instances
        input_stats["completed_inputs"] = len(records["image"])

    prediction = empty_prediction(detector_cache)
    diagnostic = {
        "inlier": 0.0,
        "precision": 0.0,
        "information": 0.0,
        "solver_weight": 0.0,
        "solver_gate": 0.0,
        "solver_condition_log10": 0.0,
        "solver_residual_rms": 0.0,
        "count": 0,
    }
    use_amp = cfg.device.startswith("cuda")
    with torch.no_grad():
        for start in tqdm(
            range(0, len(records["image"]), cfg.batch_size),
            desc="Stage2 information-field REAL275",
        ):
            end = min(start + cfg.batch_size, len(records["image"]))
            relative = torch.as_tensor(
                np.asarray(records["partial_relative"][start:end]),
                dtype=torch.float32,
                device=cfg.device,
            )
            centers = torch.as_tensor(
                np.asarray(records["alignment_center"][start:end]),
                dtype=torch.float32,
                device=cfg.device,
            )
            class_ids = torch.as_tensor(
                records["class_id"][start:end], dtype=torch.long, device=cfg.device
            )
            points = relative + centers[:, None]
            normals, observation = estimate_camera_observation_features(
                points, centers, k=cfg.normals_k
            )
            symmetric = continuous_y_symmetry_mask(class_ids, None)
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=use_amp
            ):
                output = model(
                    points,
                    normals,
                    observation,
                    alignment_center=centers,
                    continuous_symmetry=symmetric,
                )
            write_batch(prediction, records, start, output["pose"], output["size"])
            count = end - start
            diagnostic["count"] += count
            diagnostic["inlier"] += float(torch.sigmoid(output["inlier_logits"]).mean()) * count
            diagnostic["precision"] += float(output["axis_precision"].mean()) * count
            diagnostic["information"] += float(output["information"].mean()) * count
            diagnostic["solver_weight"] += (
                float(output["solver_weights"].mean()) * count
            )
            diagnostic["solver_gate"] += float(output["solver_gate"].mean()) * count
            diagnostic["solver_condition_log10"] += float(
                torch.log10(output["solver_condition"].clamp_min(1.0)).mean()
            ) * count
            diagnostic["solver_residual_rms"] += float(
                output["solver_residual_rms"].mean()
            ) * count

    prediction_path = os.path.join(cfg.output_dir, "stage2_predictions.pkl")
    with open(prediction_path, "wb") as stream:
        pickle.dump(prediction, stream, protocol=pickle.HIGHEST_PROTOCOL)
    metrics = compute_real275_3d_map(
        [item["result"] for item in prediction.values()],
        os.path.join(cfg.output_dir, "official_metrics_3d"),
    )
    values = summary_from_metrics(metrics)
    classes = per_class_diagnostics(metrics)
    count = max(1, diagnostic.pop("count"))
    values.update(
        {
            "protocol": "REAL275_NOCS_3D_IOU_AND_POSE_AP",
            "method": CHECKPOINT_FORMAT,
            "stage": "stage2_observability_guided_pose_information_field",
            "full_evaluation": cfg.max_instances == 0,
            "uses_real275_gt_for_prediction": False,
            "input": "detector_partial_absolute_camera_point_cloud_with_normals_and_view_evidence",
            "input_stats": input_stats,
            "checkpoint": os.path.abspath(cfg.checkpoint),
            "checkpoint_sha256": file_sha256(cfg.checkpoint),
            "checkpoint_epoch": checkpoint_epoch_number(state),
            "evaluated_weights": state["resolved_evaluation_weights"],
            "fps_backend": resolved_fps_backend,
            "prediction": os.path.abspath(prediction_path),
            "endpoint_count": 1,
            "single_deterministic_pose": True,
            "train_updates": training_irls_iterations,
            "test_updates": evaluation_irls_iterations,
            "training_irls_iterations": training_irls_iterations,
            "evaluation_irls_iterations": evaluation_irls_iterations,
            "final_pose_solver": restored_solver_contract,
            "generative_completion": False,
            "latent_prediction": False,
            "external_pretrained_weights": False,
            "stage1_at_inference": False,
            "category_network_input": False,
            "evaluation_batch_size": cfg.batch_size,
            "per_class_diagnostics": classes,
            "per_category_reference": compare_per_category_reference(classes),
            "solver_diagnostics": {
                key: value / count for key, value in diagnostic.items()
            },
        }
    )
    summary_path = os.path.join(cfg.output_dir, "evaluation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as stream:
        json.dump(values, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(values, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
