"""Audit whether a Stage1 checkpoint provides stable regional observability.

The audit uses CAMERA/Real training samples only. It removes local regions
from complete clouds twice with different deterministic replacement offsets,
then checks that the resulting pose/box decision changes are repeatable and
spatially non-uniform. The output JSON is bound to the exact checkpoint SHA256
and is required when that checkpoint supervises Stage2 training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os

import torch

from shared import point_ops
from shared.dataset import get_data_loaders, process_batch
from shared.point_features import estimate_point_normals
from shared.pose_geometry import pose6d_to_matrix, quotient_rotation_distance
from shared.training import NO_MASK_DEFORM, NO_ZOOM, seed_everything
from stage1.evaluate_real275 import load_model
from stage2.observability import regional_stage1_observability


CHANNELS = ("up", "yaw", "translation", "size")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/stage1.pth")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--source", choices=("CAMERA", "Real", "CAMERA+Real"), default="CAMERA+Real"
    )
    parser.add_argument("--examples", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--normals_k", type=int, default=16)
    parser.add_argument(
        "--point_ops_backend",
        choices=(point_ops.LEGACY_FPS_BACKEND, point_ops.TORCH_FPS_BACKEND),
        default=point_ops.LEGACY_FPS_BACKEND,
    )
    parser.add_argument("--regions", type=int, default=10)
    parser.add_argument("--removal_fraction", type=float, default=0.125)
    parser.add_argument("--teacher_chunk", type=int, default=24)
    parser.add_argument("--minimum_repeatability", type=float, default=0.55)
    parser.add_argument("--minimum_spatial_cv", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=260905)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pearson(first: torch.Tensor, second: torch.Tensor) -> float:
    first = first.float().reshape(-1)
    second = second.float().reshape(-1)
    first = first - first.mean()
    second = second - second.mean()
    denominator = torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)
    if float(denominator) <= 1e-12:
        return 0.0
    return float((first * second).sum() / denominator)


def decision_error(
    pose: torch.Tensor,
    size: torch.Tensor,
    target_pose: torch.Tensor,
    target_size: torch.Tensor,
    class_ids: torch.Tensor,
    handles: torch.Tensor,
) -> torch.Tensor:
    rotation = quotient_rotation_distance(
        pose6d_to_matrix(pose[..., :6]),
        pose6d_to_matrix(target_pose[..., :6]),
        class_ids,
        handles,
    ) / math.radians(5.0)
    translation = torch.linalg.vector_norm(
        pose[..., 6:] - target_pose[..., 6:], dim=-1
    ) / 0.02
    size_error = torch.linalg.vector_norm(
        torch.log(size.clamp_min(1e-5) / target_size.clamp_min(1e-5)), dim=-1
    ) / 0.10
    return rotation + translation + size_error


def main() -> None:
    cfg = arguments()
    os.environ["INFLUPOSE_FPS_BACKEND"] = cfg.point_ops_backend
    if cfg.examples <= 0 or cfg.batch_size <= 0:
        raise ValueError("examples and batch_size must be positive")
    if cfg.num_points != 1024:
        raise ValueError("the retained Stage1 checkpoint uses 1024 points")
    if cfg.regions < 2:
        raise ValueError("regions must be at least two")
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    seed_everything(cfg.seed)
    teacher, state = load_model(cfg.checkpoint, cfg.device)
    loader = get_data_loaders(
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        dynamic_zoom_in_params=NO_ZOOM,
        deform_2d_params=NO_MASK_DEFORM,
        percentage_data=1.0,
        data_path=cfg.data_path,
        source=cfg.source,
        mode="train",
        n_pts=cfg.num_points,
        img_size=256,
        num_workers=cfg.num_workers,
        pointcloud_only=True,
        camera_real_ratio=3.0,
    )

    first_impacts: list[torch.Tensor] = []
    second_impacts: list[torch.Tensor] = []
    damage_values: list[torch.Tensor] = []
    collected = 0
    for raw in loader:
        remaining = cfg.examples - collected
        if remaining <= 0:
            break
        batch = process_batch(
            raw,
            device=cfg.device,
            pose_mode="rot_matrix",
            mini_batch_size=min(remaining, len(raw["path"])),
            PTS_AUG_PARAMS=None,
            return_full_cloud=True,
            full_cloud_use_dense=False,
        )
        points = batch["full_cloud"].float()
        normals = estimate_point_normals(
            points, points.mean(dim=1), k=cfg.normals_k
        ).detach()
        centers = batch["pts_center"].float()
        class_ids = batch["id"].long()
        handles = batch["handle_visibility"].long()
        target_pose = batch["gt_pose"].float()
        target_size = batch["gt_size"].float()

        first = regional_stage1_observability(
            teacher,
            points,
            normals,
            centers,
            class_ids,
            handles,
            region_count=cfg.regions,
            removal_fraction=cfg.removal_fraction,
            repeat_offset=0,
            teacher_chunk=cfg.teacher_chunk,
        )
        second = regional_stage1_observability(
            teacher,
            points,
            normals,
            centers,
            class_ids,
            handles,
            region_count=cfg.regions,
            removal_fraction=cfg.removal_fraction,
            repeat_offset=1,
            teacher_chunk=cfg.teacher_chunk,
        )
        first_impacts.append(first["raw_impact"].cpu())
        second_impacts.append(second["raw_impact"].cpu())

        base_error = decision_error(
            first["baseline_pose"],
            first["baseline_size"],
            target_pose,
            target_size,
            class_ids,
            handles,
        )
        region_count = first["ablated_pose"].shape[1]
        expanded_pose = target_pose[:, None].expand(-1, region_count, -1)
        expanded_size = target_size[:, None].expand(-1, region_count, -1)
        ablated_error = decision_error(
            first["ablated_pose"],
            first["ablated_size"],
            expanded_pose,
            expanded_size,
            class_ids,
            handles,
        )
        damage_values.append((ablated_error - base_error[:, None]).clamp_min(0.0).cpu())
        collected += points.shape[0]

    if collected < cfg.examples:
        raise RuntimeError(f"dataset yielded only {collected} audit examples")

    first = torch.cat(first_impacts, dim=0)[: cfg.examples]
    second = torch.cat(second_impacts, dim=0)[: cfg.examples]
    damage = torch.cat(damage_values, dim=0)[: cfg.examples]
    repeatability = {
        name: pearson(first[..., index], second[..., index])
        for index, name in enumerate(CHANNELS)
    }
    active_repeatability = [
        value
        for index, value in enumerate(repeatability.values())
        if float(first[..., index].std()) > 1e-8
    ]
    median_repeatability = float(torch.tensor(active_repeatability).median())

    spatial_mean = first.mean(dim=1)
    spatial_cv = first.std(dim=1, unbiased=False) / spatial_mean.clamp_min(1e-6)
    active_cv = spatial_cv[spatial_mean > 1e-6]
    median_spatial_cv = float(active_cv.median()) if active_cv.numel() else 0.0

    impact_score = first.sum(dim=-1)
    impact_damage_correlation = pearson(impact_score, damage)
    top = impact_score.argmax(dim=1)
    bottom = impact_score.argmin(dim=1)
    batch_indices = torch.arange(impact_score.shape[0])
    top_minus_bottom = float(
        (damage[batch_indices, top] - damage[batch_indices, bottom]).mean()
    )

    finite = bool(
        torch.isfinite(first).all()
        and torch.isfinite(second).all()
        and torch.isfinite(damage).all()
    )
    passed = bool(
        finite
        and min(active_repeatability, default=0.0) >= cfg.minimum_repeatability
        and median_spatial_cv >= cfg.minimum_spatial_cv
    )
    flattened = first.reshape(-1, len(CHANNELS))
    report = {
        "format": "STAGE1_OCCLUSION_OBSERVABILITY_AUDIT_V1",
        "passed": passed,
        "checkpoint": os.path.abspath(cfg.checkpoint),
        "checkpoint_sha256": file_sha256(cfg.checkpoint),
        "checkpoint_epoch": int(state.get("epoch", -1)) + 1,
        "examples": cfg.examples,
        "regions_per_example": cfg.regions,
        "removal_fraction": cfg.removal_fraction,
        "uses_real275_test": False,
        "uses_rgb": False,
        "category_network_input": False,
        "gate": {
            "finite": finite,
            "minimum_repeatability": cfg.minimum_repeatability,
            "minimum_spatial_cv": cfg.minimum_spatial_cv,
        },
        "repeatability_pearson": repeatability,
        "median_active_repeatability": median_repeatability,
        "median_spatial_coefficient_of_variation": median_spatial_cv,
        "impact_vs_gt_damage_correlation_diagnostic": impact_damage_correlation,
        "top_minus_bottom_region_gt_damage": top_minus_bottom,
        "raw_impact_mean": {
            name: float(flattened[:, index].mean())
            for index, name in enumerate(CHANNELS)
        },
        "raw_impact_p95": {
            name: float(torch.quantile(flattened[:, index], 0.95))
            for index, name in enumerate(CHANNELS)
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(cfg.output)), exist_ok=True)
    with open(cfg.output, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(report, sort_keys=True), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
