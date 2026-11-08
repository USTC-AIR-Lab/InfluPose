"""Frozen Stage1 regional-ablation labels for Stage2 observability learning."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from shared import point_ops
from shared.pose_geometry import (
    continuous_y_symmetry_mask,
    pose6d_to_matrix,
    so3_log_map,
)


@torch.no_grad()
def teacher_decision(
    teacher,
    points: torch.Tensor,
    normals: torch.Tensor,
    alignment_center: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Decode the deterministic posterior mean of frozen complete-cloud Stage1."""

    mean, _, _, _, full_center, _, _ = teacher.encode_distribution(
        points,
        alignment_center=alignment_center,
        normals=normals,
    )
    return teacher.decode_latent(
        mean,
        alignment_center=alignment_center,
        full_cloud_center=full_center,
    )


def _gather_regions(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    expanded = values[:, None].expand(-1, indices.shape[1], -1, -1)
    return torch.gather(
        expanded,
        2,
        indices[..., None].expand(-1, -1, -1, values.shape[-1]),
    )


@torch.no_grad()
def regional_ablation_plan(
    full_points: torch.Tensor,
    *,
    region_count: int = 8,
    removal_fraction: float = 0.125,
    repeat_offsets: tuple[int, ...] = (0,),
) -> dict[str, torch.Tensor]:
    """Freeze region centers and replacement indices for teacher comparisons.

    A multi-checkpoint audit must not rerun FPS independently for every
    checkpoint: even a numerically equivalent region sampler would leave a
    needless ambiguity about whether two teachers saw exactly the same local
    interventions.  This helper builds the intervention plan once.  The plan
    can then be passed to :func:`regional_stage1_observability` for every
    checkpoint and is small enough to archive with the audit tensors.
    """

    if full_points.ndim != 3 or full_points.shape[-1] != 3:
        raise ValueError("full_points must have shape [B,N,3]")
    if not 2 <= region_count <= full_points.shape[1]:
        raise ValueError("region_count lies outside the point cloud")
    if not 0.01 <= removal_fraction <= 0.50:
        raise ValueError("removal_fraction must lie in [0.01,0.50]")
    if not repeat_offsets:
        raise ValueError("at least one repeat offset is required")
    offsets = tuple(int(value) for value in repeat_offsets)
    if len(set(offsets)) != len(offsets):
        raise ValueError("repeat offsets must be distinct")

    _, point_count, _ = full_points.shape
    center = full_points.mean(dim=1, keepdim=True)
    extent = (full_points.amax(dim=1) - full_points.amin(dim=1)).amax(
        dim=-1, keepdim=True
    ).clamp_min(0.03)
    normalized = (full_points - center) / extent[:, None]
    region_indices = point_ops.farthest_point_sample(
        normalized.contiguous(), region_count
    ).long()
    region_centers = torch.gather(
        full_points,
        1,
        region_indices[..., None].expand(-1, -1, 3),
    )
    distance = torch.cdist(region_centers, full_points)
    remove_count = max(1, round(point_count * removal_fraction))
    keep_count = point_count - remove_count
    keep_indices = distance.topk(keep_count, dim=-1, largest=True).indices
    plans: list[torch.Tensor] = []
    for offset in offsets:
        duplicate_source = torch.roll(
            keep_indices,
            shifts=(offset * max(1, remove_count // 3)) % keep_count,
            dims=-1,
        )[..., :remove_count]
        plans.append(torch.cat((keep_indices, duplicate_source), dim=-1))
    return {
        "region_indices": region_indices,
        "region_centers": region_centers,
        "ablated_indices": torch.stack(plans, dim=0),
        "repeat_offsets": torch.tensor(
            offsets, dtype=torch.long, device=full_points.device
        ),
        "removed_points": torch.tensor(
            remove_count, dtype=torch.long, device=full_points.device
        ),
        "point_count": torch.tensor(
            point_count, dtype=torch.long, device=full_points.device
        ),
    }


def _select_ablation_plan(
    full_points: torch.Tensor,
    plan: dict[str, torch.Tensor],
    *,
    region_count: int,
    removal_fraction: float,
    repeat_offset: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    required = {
        "region_indices",
        "ablated_indices",
        "repeat_offsets",
        "removed_points",
        "point_count",
    }
    missing = required.difference(plan)
    if missing:
        raise ValueError(f"ablation plan is missing fields: {sorted(missing)}")
    batch, point_count, _ = full_points.shape
    region_indices = plan["region_indices"].long()
    ablated = plan["ablated_indices"].long()
    offsets = plan["repeat_offsets"].long().reshape(-1)
    if region_indices.shape != (batch, region_count):
        raise ValueError("ablation plan region indices do not align")
    if ablated.shape != (offsets.numel(), batch, region_count, point_count):
        raise ValueError("ablation plan replacement indices do not align")
    if int(plan["point_count"].item()) != point_count:
        raise ValueError("ablation plan was built for another point count")
    expected_removed = max(1, round(point_count * removal_fraction))
    if int(plan["removed_points"].item()) != expected_removed:
        raise ValueError("ablation plan uses another removal fraction")
    if (
        torch.any(region_indices < 0)
        or torch.any(region_indices >= point_count)
        or torch.any(ablated < 0)
        or torch.any(ablated >= point_count)
    ):
        raise ValueError("ablation plan contains an out-of-range point index")
    matches = torch.nonzero(offsets == int(repeat_offset), as_tuple=False).reshape(-1)
    if matches.numel() != 1:
        raise ValueError("requested repeat offset is absent from the ablation plan")
    region_centers = torch.gather(
        full_points,
        1,
        region_indices[..., None].expand(-1, -1, 3),
    )
    return (
        region_indices,
        region_centers,
        ablated[int(matches.item())],
        plan["removed_points"],
    )


@torch.no_grad()
def regional_stage1_observability(
    teacher,
    full_points: torch.Tensor,
    full_normals: torch.Tensor,
    alignment_center: torch.Tensor,
    class_ids: torch.Tensor,
    handles: torch.Tensor,
    *,
    region_count: int = 8,
    removal_fraction: float = 0.125,
    repeat_offset: int = 0,
    teacher_chunk: int = 24,
    ablation_plan: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Measure four Stage1 decision changes after removing local CAD regions.

    Removed points are replaced only by points outside the selected region so
    the frozen encoder keeps its 1024-point input contract. ``repeat_offset``
    changes which retained points are duplicated and is used to audit whether
    the measured response is a stable region effect rather than a resampling
    artifact.
    """

    if full_points.ndim != 3 or full_points.shape[-1] != 3:
        raise ValueError("full_points must have shape [B,N,3]")
    if full_normals.shape != full_points.shape:
        raise ValueError("full_normals must align with full_points")
    if not 2 <= region_count <= full_points.shape[1]:
        raise ValueError("region_count lies outside the point cloud")
    if not 0.01 <= removal_fraction <= 0.50:
        raise ValueError("removal_fraction must lie in [0.01,0.50]")
    if teacher_chunk <= 0:
        raise ValueError("teacher_chunk must be positive")

    batch, point_count, _ = full_points.shape
    if ablation_plan is None:
        ablation_plan = regional_ablation_plan(
            full_points,
            region_count=region_count,
            removal_fraction=removal_fraction,
            repeat_offsets=(repeat_offset,),
        )
    region_indices, region_centers, ablated_indices, removed_points = (
        _select_ablation_plan(
            full_points,
            ablation_plan,
            region_count=region_count,
            removal_fraction=removal_fraction,
            repeat_offset=repeat_offset,
        )
    )
    ablated_points = _gather_regions(full_points, ablated_indices)
    ablated_normals = _gather_regions(full_normals, ablated_indices)

    baseline = teacher_decision(
        teacher, full_points, full_normals, alignment_center
    )
    flat_points = ablated_points.reshape(batch * region_count, point_count, 3)
    flat_normals = ablated_normals.reshape(batch * region_count, point_count, 3)
    flat_center = alignment_center[:, None].expand(-1, region_count, -1).reshape(
        batch * region_count, 3
    )
    pose_parts: list[torch.Tensor] = []
    size_parts: list[torch.Tensor] = []
    for start in range(0, flat_points.shape[0], teacher_chunk):
        end = min(start + teacher_chunk, flat_points.shape[0])
        decoded = teacher_decision(
            teacher,
            flat_points[start:end],
            flat_normals[start:end],
            flat_center[start:end],
        )
        pose_parts.append(decoded["pose"].float())
        size_parts.append(decoded["size"].float())
    ablated_pose = torch.cat(pose_parts).reshape(batch, region_count, 9)
    ablated_size = torch.cat(size_parts).reshape(batch, region_count, 3)

    base_rotation = pose6d_to_matrix(baseline["pose"][:, :6].float())
    ablated_rotation = pose6d_to_matrix(ablated_pose[..., :6].reshape(-1, 6))
    ablated_rotation = ablated_rotation.reshape(batch, region_count, 3, 3)
    relative = ablated_rotation @ base_rotation[:, None].transpose(-1, -2)
    tangent = so3_log_map(relative.reshape(-1, 3, 3), eps=1e-4).reshape(
        batch, region_count, 3
    )
    tangent = torch.nan_to_num(tangent, 0.0)
    base_up = F.normalize(base_rotation[:, :, 1], dim=-1, eps=1e-6)
    yaw_signed = (tangent * base_up[:, None]).sum(dim=-1)
    yaw = yaw_signed.abs()
    tilt = torch.linalg.vector_norm(
        tangent - yaw_signed[..., None] * base_up[:, None], dim=-1
    )
    symmetric = continuous_y_symmetry_mask(class_ids, handles)
    yaw = torch.where(symmetric[:, None], torch.zeros_like(yaw), yaw)
    translation = torch.linalg.vector_norm(
        ablated_pose[..., 6:] - baseline["pose"][:, None, 6:].float(), dim=-1
    ) / 0.02
    size = torch.linalg.vector_norm(
        torch.log(
            ablated_size
            / baseline["size"][:, None].float().clamp_min(1e-5)
        ),
        dim=-1,
    ) / 0.10
    raw = torch.stack((tilt, yaw, translation, size), dim=-1)
    raw = torch.nan_to_num(raw, 0.0).clamp_min(0.0)
    normalizer = raw.mean(dim=1, keepdim=True)
    normalized_impact = torch.where(
        normalizer > 1e-6,
        raw / normalizer.clamp_min(1e-6),
        torch.zeros_like(raw),
    ).clamp_max(10.0)
    return {
        "region_centers": region_centers,
        "region_indices": region_indices,
        "raw_impact": raw,
        "normalized_impact": normalized_impact,
        "baseline_pose": baseline["pose"].float(),
        "baseline_size": baseline["size"].float(),
        "ablated_pose": ablated_pose,
        "ablated_size": ablated_size,
        "removed_points": removed_points,
    }


def transfer_region_information(
    anchor_xyz: torch.Tensor,
    region_centers: torch.Tensor,
    region_information: torch.Tensor,
) -> torch.Tensor:
    """Nearest-region interpolation of ablation labels onto visible anchors."""

    if anchor_xyz.ndim != 3 or anchor_xyz.shape[-1] != 3:
        raise ValueError("anchor_xyz must have shape [B,K,3]")
    if region_centers.shape[:1] != anchor_xyz.shape[:1]:
        raise ValueError("region centers do not align with anchor batch")
    if region_information.shape != (*region_centers.shape[:2], 4):
        raise ValueError("region_information must have shape [B,R,4]")
    nearest = torch.cdist(anchor_xyz, region_centers).argmin(dim=-1)
    return torch.gather(
        region_information,
        1,
        nearest[..., None].expand(-1, -1, 4),
    )


def analytic_correspondence_information(
    canonical: torch.Tensor,
    size: torch.Tensor,
    symmetric: torch.Tensor,
) -> torch.Tensor:
    """Geometry-only Fisher proxy used when an online Stage1 label is absent."""

    metric = canonical * size[:, None]
    radial = torch.linalg.vector_norm(metric[..., (0, 2)], dim=-1)
    vertical = metric[..., 1].abs()
    up = torch.sqrt(vertical.square() + 0.25 * radial.square()).clamp_min(1e-4)
    yaw = radial.clamp_min(1e-4)
    yaw = torch.where(symmetric[:, None], torch.zeros_like(yaw), yaw)
    translation = torch.ones_like(up)
    size_information = torch.linalg.vector_norm(canonical, dim=-1).clamp_min(1e-4)
    values = torch.stack((up, yaw, translation, size_information), dim=-1)
    mean = values.mean(dim=1, keepdim=True)
    return torch.where(
        mean > 1e-6,
        values / mean.clamp_min(1e-6),
        torch.zeros_like(values),
    ).clamp(0.0, 10.0)
