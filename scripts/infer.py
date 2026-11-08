"""Estimate object poses and metric sizes from partial point clouds."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.point_features import estimate_camera_observation_features
from shared.pose_geometry import continuous_y_symmetry_mask, pose6d_to_matrix
from stage2.evaluate_real275 import load_model


def predict_frame(model, points, class_ids, device, batch_size=48):
    """Recover pose and size with one model forward per detected object."""
    points = np.asarray(points, dtype=np.float32)
    class_ids = np.asarray(class_ids)
    count = len(points)
    if points.shape != (count, 1024, 3) or not np.isfinite(points).all():
        raise ValueError("points must be finite camera-frame XYZ, shape (M, 1024, 3)")
    if class_ids.shape != (count,) or not np.isin(class_ids, np.arange(1, 7)).all():
        raise ValueError("class_ids must have shape (M,) with NOCS IDs 1 through 6")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    transforms = np.repeat(np.eye(4, dtype=np.float64)[None], count, axis=0)
    sizes = np.empty((count, 3), dtype=np.float64)
    model.irls_iterations = 1
    with torch.no_grad():
        for start in range(0, count, batch_size):
            end = min(start + batch_size, count)
            clouds = points[start:end]
            center_array = clouds.mean(axis=1)
            centers = torch.as_tensor(center_array, dtype=torch.float32, device=device)
            relative = torch.as_tensor(clouds - center_array[:, None], device=device)
            absolute = relative + centers[:, None]
            detected_classes = torch.as_tensor(class_ids[start:end] - 1, dtype=torch.long, device=device)
            normals, observation = estimate_camera_observation_features(absolute, centers, k=16)
            symmetric = continuous_y_symmetry_mask(detected_classes, None)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                output = model(absolute, normals, observation, alignment_center=centers,
                               continuous_symmetry=symmetric)
            transforms[start:end, :3, :3] = pose6d_to_matrix(output["pose"][:, :6]).cpu().numpy()
            transforms[start:end, :3, 3] = output["pose"][:, 6:].cpu().numpy()
            sizes[start:end] = output["size"].cpu().numpy()
    return transforms, sizes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--checkpoint", default="checkpoints/stage2.pth")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backend", choices=("legacy_cuda", "torch"), default="legacy_cuda")
    cfg = parser.parse_args()
    os.environ["INFLUPOSE_FPS_BACKEND"] = cfg.backend
    np.random.seed(260925)
    torch.manual_seed(260925)
    if cfg.device.startswith("cuda"):
        torch.cuda.manual_seed_all(260925)
    with np.load(cfg.observations, allow_pickle=False) as frame:
        points, class_ids = frame["points"], frame["class_ids"]
    model, _ = load_model(cfg.checkpoint, cfg.device, "ema")
    transforms, sizes = predict_frame(model, points, class_ids, cfg.device)
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cfg.output, rotation=transforms[:, :3, :3], translation=transforms[:, :3, 3], size=sizes)
    print(str(cfg.output))


if __name__ == "__main__":
    main()
