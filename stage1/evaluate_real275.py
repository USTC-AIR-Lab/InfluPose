"""Evaluate the Stage1 complete-cloud upper bound on REAL275."""

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
from shared.point_features import estimate_point_normals
from shared.pose_geometry import pose6d_to_matrix
from shared.real275_io import empty_prediction, oracle_complete_cloud_records
from shared.real275_metrics import (
    compute_real275_3d_map,
    per_class_diagnostics,
    summary_from_metrics,
)
from stage1.model import Stage1CompleteCloudRAE


from stage1.checkpoint import CHECKPOINT_FORMAT as CHECKPOINT_FORMAT, load_model as load_model


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/stage1.pth")
    parser.add_argument(
        "--detector_cache",
        default="data/evaluation/segmentation_results_real_test.pkl",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--normals_k", type=int, default=16)
    parser.add_argument(
        "--point_ops_backend",
        choices=(point_ops.LEGACY_FPS_BACKEND, point_ops.TORCH_FPS_BACKEND),
        default=point_ops.LEGACY_FPS_BACKEND,
        help="use the archived CUDA primitives for exact Stage1/G10A inference",
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()




@torch.no_grad()
def predict(
    model: Stage1CompleteCloudRAE,
    cache: dict,
    records: dict[str, list],
    *,
    batch_size: int,
    normals_k: int,
    device: str,
) -> dict:
    output = empty_prediction(cache)
    for start in tqdm(
        range(0, len(records["points"]), batch_size), desc="Stage1 REAL275 upper bound"
    ):
        end = min(start + batch_size, len(records["points"]))
        points = torch.as_tensor(
            np.asarray(records["points"][start:end]),
            dtype=torch.float32,
            device=device,
        )
        centers = torch.as_tensor(
            np.asarray(records["alignment_center"][start:end]),
            dtype=torch.float32,
            device=device,
        )
        normals = estimate_point_normals(points, points.mean(dim=1), k=normals_k)
        prediction = model(
            points,
            alignment_center=centers,
            normals=normals,
            reconstruction_count=0,
        )
        rotations = pose6d_to_matrix(prediction["pose"][:, :6]).cpu().numpy()
        translations = prediction["pose"][:, 6:].cpu().numpy()
        sizes = prediction["size"].cpu().numpy()
        for offset in range(end - start):
            image_path = records["image"][start + offset]
            prediction_index = records["prediction_index"][start + offset]
            transform = np.eye(4, dtype=np.float64)
            transform[:3, :3] = rotations[offset]
            transform[:3, 3] = translations[offset]
            output[image_path]["result"]["pred_RTs"][prediction_index] = transform
            output[image_path]["result"]["pred_scales"][prediction_index] = sizes[offset]
    return output


def main() -> None:
    cfg = arguments()
    os.environ["INFLUPOSE_FPS_BACKEND"] = cfg.point_ops_backend
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(cfg.detector_cache, "rb") as stream:
        cache = pickle.load(stream)
    records, input_stats = oracle_complete_cloud_records(cache, cfg.data_path)
    model, state = load_model(cfg.checkpoint, cfg.device)
    predictions = predict(
        model,
        cache,
        records,
        batch_size=cfg.batch_size,
        normals_k=cfg.normals_k,
        device=cfg.device,
    )
    prediction_path = os.path.join(cfg.output_dir, "predictions.pkl")
    with open(prediction_path, "wb") as stream:
        pickle.dump(predictions, stream, protocol=pickle.HIGHEST_PROTOCOL)
    metrics = compute_real275_3d_map(
        [item["result"] for item in predictions.values()],
        os.path.join(cfg.output_dir, "official_metrics_3d"),
    )
    values = summary_from_metrics(metrics)
    values.update(
        {
            "protocol": "REAL275_NOCS_3D_IOU_AND_POSE_AP",
            "stage": "stage1_complete_cloud_rae128",
            "method": CHECKPOINT_FORMAT,
            "input_mode": "oracle_complete_cloud",
            "input_stats": input_stats,
            "checkpoint": os.path.abspath(cfg.checkpoint),
            "checkpoint_sha256": file_sha256(cfg.checkpoint),
            "latent_contract": state.get("latent_contract"),
            "class_prior_audit": state.get("class_prior_audit"),
            "prediction": os.path.abspath(prediction_path),
            "per_class_diagnostics": per_class_diagnostics(metrics),
        }
    )
    summary_path = os.path.join(cfg.output_dir, "evaluation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as stream:
        json.dump(values, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(values, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
