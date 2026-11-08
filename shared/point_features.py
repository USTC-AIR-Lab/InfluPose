"""Cheap per-point local geometry features for the decision encoder.

``estimate_point_normals`` computes local surface normals from k-nearest
neighbour covariance PCA.  Normals are per-point, local, and rotate with the
cloud (rotation-equivariant), giving the retained point Transformers the local
direction cues needed by camera and mug instances.
"""

from __future__ import annotations

import torch


def _local_covariance(points: torch.Tensor, center: torch.Tensor, k: int, chunk: int):
    """Return covariance [C,N,3,3] and kNN index for a sub-batch."""
    batch, num_points, _ = points.shape
    if num_points <= k + 1:
        k = max(1, num_points - 1)
    normals_out = torch.empty(points.shape[0], num_points, 3, device=points.device)
    curvature = torch.empty(points.shape[0], num_points, 3, device=points.device)
    for start in range(0, batch, chunk):
        end = min(batch, start + chunk)
        sub = points[start:end]
        distance = torch.cdist(sub, sub)
        _, index = distance.topk(k + 1, dim=-1, largest=False)
        index = index[..., 1:]
        # ``index[b, i, :]`` indexes the point-set axis of ``sub``.  The old
        # layout repeated ``sub[b, i]`` along that axis, so it gathered point
        # ``i`` as every neighbour and made every covariance identically
        # zero.  Keep queries in dimension 1 and candidates in dimension 2.
        candidates = sub[:, None, :, :].expand(-1, num_points, -1, -1)
        neighbor = torch.gather(
            candidates, 2, index[..., None].expand(-1, -1, -1, 3)
        )
        centered = neighbor - sub[:, :, None, :]
        covariance = centered.transpose(-1, -2) @ centered / max(1, k)
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        except RuntimeError:
            identity = torch.eye(3, device=points.device, dtype=points.dtype)
            # ascending eigenvalues (eigh convention): lambda0<=lambda1<=lambda2,
            # so curvature ratios stay finite on degenerate neighbourhoods.
            eigenvalues = torch.tensor(
                [0.0, 0.0, 1.0], device=points.device, dtype=points.dtype
            ).expand(covariance.shape[0], covariance.shape[1], 3).clone()
            eigenvectors = identity[None, None].expand(
                covariance.shape[0], covariance.shape[1], 3, 3
            ).clone()
        normal = eigenvectors[..., 0]
        outward = sub - center[start:end, None, :]
        sign = torch.einsum("bnd,bnd->bn", normal, outward)
        normal = torch.where(sign[..., None] < 0, -normal, normal)
        normals_out[start:end] = normal
        eigenvalues = eigenvalues.clamp_min(1e-8)
        l0, l1, l2 = eigenvalues[..., 0], eigenvalues[..., 1], eigenvalues[..., 2]
        curvature_out = torch.stack(
            (
                (l1 - l2) / l2,  # linearity
                (l2 - l0) / l2,  # planarity
                l0 / l2,  # scattering
            ),
            dim=-1,
        )
        curvature[start:end] = curvature_out
    return normals_out, curvature


def estimate_point_normals(
    points: torch.Tensor,
    center: torch.Tensor,
    k: int = 16,
    chunk: int = 8,
) -> torch.Tensor:
    """Estimate local surface normals [B,N,3] from [B,N,3] points."""
    points = torch.nan_to_num(points.float(), 0.0)
    center = torch.nan_to_num(center.float(), 0.0)
    normals, _ = _local_covariance(points, center, k, chunk)
    return normals


def estimate_geometry_features(
    points: torch.Tensor,
    center: torch.Tensor,
    k: int = 32,
    chunk: int = 8,
) -> torch.Tensor:
    """Per-point local geometry: normals + linearity/planarity/scattering.

    All six channels are rotation-equivariant / invariant, giving the custom
    encoder both direction (normals) and shape (curvature ratios) cues.
    """
    points = torch.nan_to_num(points.float(), 0.0)
    center = torch.nan_to_num(center.float(), 0.0)
    normals, curvature = _local_covariance(points, center, k, chunk)
    return torch.cat((normals, curvature), dim=-1)


def estimate_camera_observation_features(
    points: torch.Tensor,
    center: torch.Tensor,
    k: int = 16,
    chunk: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normals and partial-view evidence for camera-frame point clouds.

    The complete-cloud Stage1 encoder only needs surface normals.  A partial
    observation additionally needs to know *how* a surface was observed:
    camera ray, incidence, local sampling density, and depth discontinuity are
    useful cues for thin laptop planes, camera fronts, and sparse mug handles.
    All scale-valued channels are normalized per cloud, while normals/rays
    rotate with a camera-frame SE(3) augmentation.

    Returns:
        normals: ``[B,N,3]`` outward-oriented PCA normals.
        evidence: ``[B,N,14]`` containing normals, standard curvature ratios,
            camera ray, signed/absolute incidence, relative depth, local
            density, and local depth spread.
    """

    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("points must have shape [batch, points, 3]")
    if center.shape != (points.shape[0], 3):
        raise ValueError("center must have shape [batch, 3]")
    if k <= 0 or chunk <= 0:
        raise ValueError("k and chunk must be positive")
    points = torch.nan_to_num(points.float(), 0.0)
    center = torch.nan_to_num(center.float(), 0.0)
    batch, num_points, _ = points.shape
    neighbour_count = min(k, max(1, num_points - 1))
    normals_out = torch.empty_like(points)
    evidence_out = torch.empty(
        batch, num_points, 14, device=points.device, dtype=points.dtype
    )
    identity = torch.eye(3, device=points.device, dtype=points.dtype)

    for start in range(0, batch, chunk):
        end = min(batch, start + chunk)
        sub = points[start:end]
        distance = torch.cdist(sub, sub)
        neighbour_distance, index = distance.topk(
            neighbour_count + 1, dim=-1, largest=False
        )
        neighbour_distance = neighbour_distance[..., 1:]
        index = index[..., 1:]
        candidates = sub[:, None, :, :].expand(-1, num_points, -1, -1)
        neighbour = torch.gather(
            candidates, 2, index[..., None].expand(-1, -1, -1, 3)
        )
        local = neighbour - sub[:, :, None, :]
        covariance = local.transpose(-1, -2) @ local / max(1, neighbour_count)
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        except RuntimeError:
            eigenvalues = sub.new_tensor((0.0, 0.0, 1.0)).expand(
                sub.shape[0], num_points, 3
            ).clone()
            eigenvectors = identity[None, None].expand(
                sub.shape[0], num_points, 3, 3
            ).clone()

        normal = eigenvectors[..., 0]
        outward = sub - center[start:end, None, :]
        sign = torch.einsum("bnd,bnd->bn", normal, outward)
        normal = torch.where(sign[..., None] < 0.0, -normal, normal)
        normal = torch.nn.functional.normalize(normal, dim=-1, eps=1e-6)

        eigenvalues = eigenvalues.clamp_min(1e-8)
        l0, l1, l2 = eigenvalues.unbind(dim=-1)
        curvature = torch.stack(
            (
                (l2 - l1) / l2,  # linearity
                (l1 - l0) / l2,  # planarity
                l0 / l2,         # scattering
            ),
            dim=-1,
        ).clamp(0.0, 1.0)

        relative = sub - center[start:end, None, :]
        extent = relative.amax(dim=1) - relative.amin(dim=1)
        scale = extent.amax(dim=-1, keepdim=True).clamp_min(0.03)
        ray = torch.nn.functional.normalize(sub, dim=-1, eps=1e-6)
        incidence = (normal * (-ray)).sum(dim=-1, keepdim=True)
        relative_depth = relative[..., 2:3] / scale[:, None]
        density = neighbour_distance.mean(dim=-1, keepdim=True) / scale[:, None]
        depth_spread = local[..., 2].std(
            dim=-1, keepdim=True, unbiased=False
        ) / scale[:, None]
        evidence = torch.cat(
            (
                normal,
                curvature,
                ray,
                incidence,
                incidence.abs(),
                relative_depth,
                density,
                depth_spread,
            ),
            dim=-1,
        )
        normals_out[start:end] = normal
        evidence_out[start:end] = torch.nan_to_num(evidence, 0.0)

    return normals_out, evidence_out


def feature_tensor(
    points: torch.Tensor,
    center: torch.Tensor,
    kind: str = "none",
    k: int = 16,
):
    """Entry point used by runners/evaluation / training."""
    if kind == "none":
        return None
    if kind == "normals":
        return estimate_point_normals(points, center, k=k)
    if kind == "geometry":
        return estimate_geometry_features(points, center, k=k)
    if kind == "camera_observation":
        return estimate_camera_observation_features(points, center, k=k)[1]
    raise ValueError(f"Unknown point feature kind: {kind}")
