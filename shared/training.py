"""Shared full-dataset training utilities for Stage1 and Stage2."""

from __future__ import annotations

import math
import os
import random

import torch
import torch.distributed as dist
import torch.nn.functional as F

from shared.pose_geometry import (
    axis_angle_to_matrix,
    continuous_y_symmetry_mask,
    matrix_to_pose6d,
    pose6d_to_matrix,
    quotient_rotation_distance,
)


DATA_AUGMENTATION = {
    "DZI_PAD_SCALE": 1.5,
    "DZI_TYPE": "uniform",
    "DZI_SCALE_RATIO": 0.25,
    "DZI_SHIFT_RATIO": 0.25,
}
POINT_AUGMENTATION = {
    "aug_pc_pro": 0.2,
    "aug_pc_r": 0.2,
    "aug_rt_pro": 0.3,
    "aug_bb_pro": 0.3,
    "aug_bc_pro": 0.3,
}
MASK_AUGMENTATION = {"roi_mask_r": 3, "roi_mask_pro": 0.5}
NO_ZOOM = {
    "DZI_PAD_SCALE": 1.5,
    "DZI_TYPE": "uniform",
    "DZI_SCALE_RATIO": 0.0,
    "DZI_SHIFT_RATIO": 0.0,
}
NO_MASK_DEFORM = {"roi_mask_r": 3, "roi_mask_pro": 0.0}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_distributed(cfg) -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = bool(cfg.distributed or world_size > 1)
    if distributed and world_size <= 1:
        raise ValueError("--distributed requires torchrun with WORLD_SIZE > 1")
    if not distributed:
        return False, 0, 0, 1
    if not cfg.device.startswith("cuda"):
        raise ValueError("distributed training requires CUDA")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError("LOCAL_RANK exceeds the visible CUDA device count")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    cfg.device = f"cuda:{local_rank}"
    return True, rank, local_rank, world_size


def reduce_epoch_totals(
    totals: dict[str, float], examples: int, *, device: str, distributed: bool
) -> tuple[dict[str, float], int]:
    if not distributed:
        return totals, examples
    keys = sorted(totals)
    packed = torch.tensor(
        [float(examples)] + [float(totals[key]) for key in keys],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    return {
        key: float(packed[index + 1]) for index, key in enumerate(keys)
    }, int(packed[0])


def select_batch(batch: dict, mask: torch.Tensor) -> dict:
    selected = {}
    batch_size = int(mask.shape[0])
    for key, value in batch.items():
        if torch.is_tensor(value) and value.ndim and value.shape[0] == batch_size:
            selected[key] = value[mask]
        else:
            selected[key] = value
    return selected


def full_dataset_selection_mask(raw: dict, cfg, *, split: str = "train") -> torch.Tensor:
    """Formal runs always train on the complete configured dataset."""

    if split != "train":
        raise ValueError("the formal trainer exposes only the full training split")
    if getattr(cfg, "real_holdout_mode", "none") != "none":
        raise ValueError("formal full-dataset training does not permit a Real holdout")
    if int(getattr(cfg, "real_holdout_percent", 0)) != 0:
        raise ValueError("formal full-dataset training requires zero holdout percent")
    count = len(raw["path"])
    return torch.ones(count, dtype=torch.bool, device=cfg.device)


def corrupt_complete_cloud(
    clean_cloud: torch.Tensor,
    *,
    max_std: float,
    clean_probability: float,
    epoch: int,
    warmup_epochs: int,
) -> torch.Tensor:
    if max_std <= 0.0 or epoch < warmup_epochs:
        return clean_cloud
    batch = clean_cloud.shape[0]
    std = torch.rand(
        batch, 1, 1, device=clean_cloud.device, dtype=clean_cloud.dtype
    ) * max_std
    clean = torch.rand(batch, 1, 1, device=clean_cloud.device) < clean_probability
    noise = torch.randn_like(clean_cloud) * std
    return clean_cloud + torch.where(clean, torch.zeros_like(noise), noise)


def augment_complete_cloud(
    complete_cloud: torch.Tensor,
    target_pose: torch.Tensor,
    anchor: torch.Tensor,
    *,
    rotation_probability: float,
    yaw_max_degrees: float,
    tilt_probability: float,
    tilt_max_degrees: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = complete_cloud.shape[0]
    device, dtype = complete_cloud.device, complete_cloud.dtype
    transform = torch.eye(3, dtype=dtype, device=device)[None].repeat(batch, 1, 1)
    rotation = pose6d_to_matrix(target_pose[:, :6])
    up = rotation[:, :, 1]
    if rotation_probability > 0.0:
        apply_yaw = torch.rand(batch, device=device) < rotation_probability
        max_yaw = torch.full(
            (batch,), math.radians(yaw_max_degrees), dtype=dtype, device=device
        )
        angle = (torch.rand(batch, device=device, dtype=dtype) * 2.0 - 1.0) * max_yaw
        angle[~apply_yaw] = 0.0
        transform = axis_angle_to_matrix(up * angle[:, None]) @ transform
        if tilt_probability > 0.0 and tilt_max_degrees > 0.0:
            apply_tilt = torch.rand(batch, device=device) < tilt_probability
            tilt_angle = torch.rand(batch, device=device, dtype=dtype) * math.radians(
                tilt_max_degrees
            )
            tilt_angle[~apply_tilt] = 0.0
            random_vector = torch.randn(batch, 3, device=device, dtype=dtype)
            tilt_axis = torch.linalg.cross(up, random_vector, dim=-1)
            tilt_axis = tilt_axis / torch.linalg.vector_norm(
                tilt_axis, dim=-1, keepdim=True
            ).clamp_min(1e-8)
            transform = axis_angle_to_matrix(tilt_axis * tilt_angle[:, None]) @ transform
    translation = target_pose[:, 6:]
    augmented_cloud = torch.bmm(complete_cloud, transform.transpose(-1, -2))
    augmented_anchor = torch.bmm(transform, anchor[:, :, None])[:, :, 0]
    augmented_rotation = torch.bmm(transform, rotation)
    augmented_translation = torch.bmm(transform, translation[:, :, None])[:, :, 0]
    augmented_pose = torch.cat(
        (matrix_to_pose6d(augmented_rotation), augmented_translation), dim=-1
    )
    return augmented_cloud, augmented_pose, augmented_anchor


def pose_box_loss(
    prediction: dict[str, torch.Tensor],
    target_pose: torch.Tensor,
    target_size: torch.Tensor,
    class_ids: torch.Tensor,
    handle_visibility: torch.Tensor,
    size_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    predicted_pose = prediction["pose"]
    rotation_distance = quotient_rotation_distance(
        pose6d_to_matrix(predicted_pose[:, :6]),
        pose6d_to_matrix(target_pose[:, :6]),
        class_ids,
        handle_visibility,
    )
    translation_loss = F.smooth_l1_loss(
        predicted_pose[:, 6:], target_pose[:, 6:], beta=0.01
    )
    size_loss = F.smooth_l1_loss(
        torch.log(prediction["size"].clamp_min(1e-6)),
        torch.log(target_size.clamp_min(1e-6)),
        beta=0.10,
    )
    loss = (
        rotation_distance.mean()
        + 2.0 * translation_loss
        + size_loss_weight * size_loss
    )
    translation_error = torch.linalg.vector_norm(
        predicted_pose[:, 6:] - target_pose[:, 6:], dim=-1
    )
    relative_size_error = torch.abs(prediction["size"] - target_size) / target_size
    return loss, {
        "rotation_deg": rotation_distance.mean().detach() * (180.0 / math.pi),
        "translation_cm": translation_error.mean().detach() * 100.0,
        "size_log": size_loss.detach(),
        "size_relative_percent": relative_size_error.mean().detach() * 100.0,
    }


def mild_complete_view(
    points: torch.Tensor, *, dropout_max: float, noise_std: float
) -> torch.Tensor:
    if not 0.0 <= dropout_max < 1.0 or noise_std < 0.0:
        raise ValueError("invalid complete-view augmentation")
    batch, count, _ = points.shape
    dropout = torch.rand(batch, 1, 1, device=points.device) * dropout_max
    mask = torch.rand(batch, count, 1, device=points.device) < dropout
    index = torch.randint(0, count, (batch, count), device=points.device)
    replacement = torch.gather(points, 1, index[..., None].expand(-1, -1, 3))
    view = torch.where(mask, replacement, points)
    return view + torch.randn_like(view) * noise_std if noise_std > 0.0 else view


def symmetric_diagonal_kl(
    first_mean: torch.Tensor,
    first_std: torch.Tensor,
    second_mean: torch.Tensor,
    second_std: torch.Tensor,
) -> torch.Tensor:
    if not (
        first_mean.shape == first_std.shape == second_mean.shape == second_std.shape
    ):
        raise ValueError("posterior tensors must align")
    first_var, second_var = first_std.square().clamp_min(1e-8), second_std.square().clamp_min(1e-8)
    difference = (first_mean - second_mean).square()
    first_to_second = 0.5 * (
        torch.log(second_var / first_var)
        + (first_var + difference) / second_var
        - 1.0
    )
    second_to_first = 0.5 * (
        torch.log(first_var / second_var)
        + (second_var + difference) / first_var
        - 1.0
    )
    return 0.5 * (first_to_second + second_to_first).mean()


def strict_pose_surrogate(
    prediction: dict[str, torch.Tensor],
    target_pose: torch.Tensor,
    class_ids: torch.Tensor,
    handles: torch.Tensor,
    *,
    rotation_temperature_degrees: float,
    translation_temperature_cm: float,
    class_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    loss, statistics = multi_threshold_pose_surrogate(
        prediction,
        target_pose,
        class_ids,
        handles,
        thresholds=(("5deg_2cm", 5.0, 2.0, 1.0),),
        rotation_temperature_degrees=rotation_temperature_degrees,
        translation_temperature_cm=translation_temperature_cm,
        class_weights=class_weights,
    )
    return loss, statistics["5deg_2cm"]


def _class_balanced_mean(
    values: torch.Tensor,
    class_ids: torch.Tensor,
    *,
    class_weights: torch.Tensor | None = None,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Average per example, then per present class.

    REAL275 reports a macro average over six categories.  This helper prevents
    the much larger CAMERA subset or repeated Real frames from silently
    defining the optimization target.  ``sample_weights`` may encode a known
    metric quotient property (for example, the number of observable rotation
    degrees of freedom); it is never passed to the network.
    """

    if values.ndim != 1 or class_ids.shape != values.shape:
        raise ValueError("balanced values and class ids must be one-dimensional")
    if class_weights is not None:
        class_weights = torch.as_tensor(
            class_weights, device=values.device, dtype=values.dtype
        )
        if class_weights.shape != (6,) or torch.any(class_weights <= 0):
            raise ValueError("class_weights must contain six positive values")
    if sample_weights is not None:
        sample_weights = torch.as_tensor(
            sample_weights, device=values.device, dtype=values.dtype
        )
        if sample_weights.shape != values.shape or torch.any(sample_weights <= 0):
            raise ValueError("sample_weights must align with values and be positive")

    class_means, present_weights = [], []
    for class_id in range(6):
        mask = class_ids == class_id
        if not torch.any(mask):
            continue
        if sample_weights is None:
            class_means.append(values[mask].mean())
        else:
            numeric = sample_weights[mask]
            class_means.append(
                (values[mask] * numeric).sum() / numeric.sum().clamp_min(1e-8)
            )
        class_weight = (
            values.new_ones(())
            if class_weights is None
            else class_weights[class_id]
        )
        # Preserve macro-category optimization while allowing a quotient-aware
        # sample weight to emphasize classes/samples with more observable
        # rotational degrees of freedom.  Without this factor, a constant
        # per-class sample weight cancels inside the normalized class mean.
        if sample_weights is not None:
            class_weight = class_weight * sample_weights[mask].mean()
        present_weights.append(class_weight)
    if not class_means:
        return values.mean()
    weights = torch.stack(present_weights)
    weights = weights / weights.sum().clamp_min(1e-8)
    return (torch.stack(class_means) * weights).sum()


def multi_threshold_pose_surrogate(
    prediction: dict[str, torch.Tensor],
    target_pose: torch.Tensor,
    class_ids: torch.Tensor,
    handles: torch.Tensor,
    *,
    thresholds: tuple[tuple[str, float, float, float], ...],
    rotation_temperature_degrees: float,
    translation_temperature_cm: float,
    class_weights: torch.Tensor | None = None,
    sample_weights: torch.Tensor | None = None,
    rotation_focus_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Smooth macro-AP surrogate for several rotation/translation thresholds.

    Each threshold is ``(name, rotation_degrees, translation_cm, weight)``.
    The joint barrier is the negative log of a factorized soft success event,
    up to constant scale.  An optional rotation-only barrier focuses learning
    on the orientation bottleneck without weakening translation supervision.
    """

    if not thresholds:
        raise ValueError("at least one pose threshold is required")
    if min(rotation_temperature_degrees, translation_temperature_cm) <= 0.0:
        raise ValueError("pose temperatures must be positive")
    if rotation_focus_weight < 0.0:
        raise ValueError("rotation_focus_weight must be non-negative")
    rotation = quotient_rotation_distance(
        pose6d_to_matrix(prediction["pose"][:, :6]),
        pose6d_to_matrix(target_pose[:, :6]),
        class_ids,
        handles,
    )
    translation = torch.linalg.vector_norm(
        prediction["pose"][:, 6:] - target_pose[:, 6:], dim=-1
    )
    rotation_temperature = math.radians(rotation_temperature_degrees)
    translation_temperature = translation_temperature_cm / 100.0
    total_weight = 0.0
    total_loss = rotation.new_zeros(())
    statistics: dict[str, torch.Tensor] = {}
    seen_names: set[str] = set()
    for name, rotation_degrees, translation_cm, weight in thresholds:
        if name in seen_names:
            raise ValueError(f"duplicate pose threshold name: {name}")
        seen_names.add(name)
        if min(rotation_degrees, translation_cm) <= 0.0 or weight < 0.0:
            raise ValueError("pose thresholds must be positive and weights non-negative")
        rotation_limit = math.radians(rotation_degrees)
        translation_limit = translation_cm / 100.0
        success = torch.sigmoid(
            (rotation_limit - rotation) / rotation_temperature
        ) * torch.sigmoid(
            (translation_limit - translation) / translation_temperature
        )
        statistics[name] = _class_balanced_mean(
            success,
            class_ids,
            class_weights=class_weights,
            sample_weights=sample_weights,
        ).detach()
        if weight == 0.0:
            continue
        rotation_barrier = rotation_temperature * F.softplus(
            (rotation - rotation_limit) / rotation_temperature
        )
        translation_barrier = translation_temperature * F.softplus(
            (translation - translation_limit) / translation_temperature
        )
        joint_barrier = rotation_barrier + 2.0 * translation_barrier
        joint_loss = _class_balanced_mean(
            joint_barrier,
            class_ids,
            class_weights=class_weights,
            sample_weights=sample_weights,
        )
        if rotation_focus_weight:
            joint_loss = joint_loss + rotation_focus_weight * _class_balanced_mean(
                rotation_barrier,
                class_ids,
                class_weights=class_weights,
                sample_weights=sample_weights,
            )
        total_loss = total_loss + weight * joint_loss
        total_weight += weight
    if total_weight <= 0.0:
        raise ValueError("at least one pose threshold weight must be positive")
    statistics["rotation_deg"] = (rotation.mean() * (180.0 / math.pi)).detach()
    statistics["translation_cm"] = (translation.mean() * 100.0).detach()
    return total_loss / total_weight, statistics


def slice_batch_prediction(
    prediction: dict[str, torch.Tensor], count: int, combined_count: int
) -> dict[str, torch.Tensor]:
    return {
        key: (
            value[:count]
            if torch.is_tensor(value)
            and value.ndim > 0
            and value.shape[0] == combined_count
            else value
        )
        for key, value in prediction.items()
    }


def symmetry_aware_box_iou_proxy(
    prediction_pose: torch.Tensor,
    prediction_size: torch.Tensor,
    target_pose: torch.Tensor,
    target_size: torch.Tensor,
    class_ids: torch.Tensor,
    handles: torch.Tensor,
) -> torch.Tensor:
    dtype, device = prediction_pose.dtype, prediction_pose.device
    angles = torch.arange(20, device=device, dtype=dtype) * (2.0 * math.pi / 20.0)
    yaw = torch.zeros(20, 3, 3, device=device, dtype=dtype)
    yaw[:, 0, 0], yaw[:, 0, 2] = torch.cos(angles), torch.sin(angles)
    yaw[:, 1, 1] = 1.0
    yaw[:, 2, 0], yaw[:, 2, 2] = -torch.sin(angles), torch.cos(angles)
    prediction_rotation = pose6d_to_matrix(prediction_pose[:, :6])
    target_rotation = pose6d_to_matrix(target_pose[:, :6])
    candidate_rotation = prediction_rotation[:, None] @ yaw[None]
    signs = prediction_pose.new_tensor(
        (
            (1, 1, 1),
            (1, 1, -1),
            (-1, 1, 1),
            (-1, 1, -1),
            (1, -1, 1),
            (1, -1, -1),
            (-1, -1, 1),
            (-1, -1, -1),
        )
    )
    prediction_local = signs[None, None] * prediction_size[:, None, None].clamp_min(1e-6) * 0.5
    prediction_corners = torch.einsum(
        "bkij,bknj->bkni", candidate_rotation, prediction_local.expand(-1, 20, -1, -1)
    ) + prediction_pose[:, None, None, 6:9]
    target_local = signs[None] * target_size[:, None].clamp_min(1e-6) * 0.5
    target_corners = torch.einsum(
        "bij,bnj->bni", target_rotation, target_local
    ) + target_pose[:, None, 6:9]
    prediction_min, prediction_max = prediction_corners.amin(dim=2), prediction_corners.amax(dim=2)
    target_min, target_max = target_corners.amin(dim=1)[:, None], target_corners.amax(dim=1)[:, None]
    intersection_extent = (
        torch.minimum(prediction_max, target_max)
        - torch.maximum(prediction_min, target_min)
    ).clamp_min(0.0)
    intersection = intersection_extent.prod(dim=-1)
    prediction_volume = (prediction_max - prediction_min).prod(dim=-1)
    target_volume = (target_max - target_min).prod(dim=-1)
    candidates = intersection / (
        prediction_volume + target_volume - intersection
    ).clamp_min(1e-9)
    symmetric = continuous_y_symmetry_mask(class_ids, handles)
    return torch.where(symmetric, candidates.max(dim=1).values, candidates[:, 0])


def directed_chamfer_squared(
    source: torch.Tensor, target: torch.Tensor, *, chunk_size: int = 128
) -> torch.Tensor:
    pieces = []
    for start in range(0, source.shape[1], chunk_size):
        squared = torch.cdist(source[:, start : start + chunk_size], target).square()
        pieces.append(squared.min(dim=-1).values)
    return torch.cat(pieces, dim=1).mean(dim=1)


def chamfer_squared(
    prediction: torch.Tensor, target: torch.Tensor, *, chunk_size: int = 128
) -> torch.Tensor:
    return 0.5 * (
        directed_chamfer_squared(prediction, target, chunk_size=chunk_size)
        + directed_chamfer_squared(target, prediction, chunk_size=chunk_size)
    )
