"""Point-cloud sampling and neighborhood operations.

The CUDA backend reproduces the sampling, gathering, grouping, and ball-query
operations used to train the released checkpoints. A PyTorch backend supports
CPU execution and numerical diagnostics."""

from __future__ import annotations

import importlib
import os

import torch


LEGACY_FPS_BACKEND = "legacy_cuda"
TORCH_FPS_BACKEND = "torch"


def fps_backend() -> str:
    """Return the explicitly selected point-operation implementation.

    The dependency-free PyTorch path is the public default.  The optional
    ``legacy_cuda`` reproduces the reference numerical
    trajectory for FPS, gather, grouping and ball query together.
    """

    backend = os.environ.get("INFLUPOSE_FPS_BACKEND", TORCH_FPS_BACKEND).strip().lower()
    if backend not in (TORCH_FPS_BACKEND, LEGACY_FPS_BACKEND):
        raise ValueError(
            "INFLUPOSE_FPS_BACKEND must be 'torch' or 'legacy_cuda', "
            f"not {backend!r}"
        )
    return backend


def _legacy_module():
    try:
        return importlib.import_module("pointnet2_utils")
    except ImportError as error:
        raise RuntimeError(
            "legacy_cuda point operations were requested, but the original "
            "PointNet++ CUDA extension is not on PYTHONPATH"
        ) from error


def _legacy_farthest_point_sample(
    points: torch.Tensor, sample_count: int
) -> torch.Tensor:
    if not points.is_cuda or points.dtype != torch.float32:
        raise TypeError("legacy_cuda FPS requires contiguous CUDA float32 points")
    module = _legacy_module()
    operator = getattr(module, "furthest_point_sample", None)
    if operator is None:
        raise RuntimeError("pointnet2_utils does not expose furthest_point_sample")
    return operator(points.contiguous(), int(sample_count)).long()


def _legacy_gather_operation(
    features: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    if not features.is_cuda or not indices.is_cuda:
        raise TypeError("legacy_cuda gather requires CUDA tensors")
    operator = getattr(_legacy_module(), "gather_operation", None)
    if operator is None:
        raise RuntimeError("pointnet2_utils does not expose gather_operation")
    return operator(features.contiguous(), indices.int().contiguous())


def _legacy_grouping_operation(
    features: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    if not features.is_cuda or not indices.is_cuda:
        raise TypeError("legacy_cuda grouping requires CUDA tensors")
    operator = getattr(_legacy_module(), "grouping_operation", None)
    if operator is None:
        raise RuntimeError("pointnet2_utils does not expose grouping_operation")
    return operator(features.contiguous(), indices.int().contiguous())


def _legacy_ball_query(
    radius: float,
    sample_count: int,
    points: torch.Tensor,
    centers: torch.Tensor,
) -> torch.Tensor:
    if not points.is_cuda or not centers.is_cuda:
        raise TypeError("legacy_cuda ball query requires CUDA tensors")
    if points.dtype != torch.float32 or centers.dtype != torch.float32:
        raise TypeError("legacy_cuda ball query requires CUDA float32 coordinates")
    operator = getattr(_legacy_module(), "ball_query", None)
    if operator is None:
        raise RuntimeError("pointnet2_utils does not expose ball_query")
    return operator(
        float(radius),
        int(sample_count),
        points.contiguous(),
        centers.contiguous(),
    ).long()


@torch.no_grad()
def farthest_point_sample(points: torch.Tensor, sample_count: int) -> torch.Tensor:
    """Deterministic iterative farthest-point sampling from point index zero."""

    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("points must have shape [B,N,3]")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    batch, count, _ = points.shape
    if count == 0:
        raise ValueError("cannot sample an empty point cloud")
    sample_count = min(int(sample_count), count)
    if fps_backend() == LEGACY_FPS_BACKEND:
        return _legacy_farthest_point_sample(points, sample_count)
    indices = torch.empty(
        batch, sample_count, dtype=torch.long, device=points.device
    )
    minimum_squared_distance = torch.full(
        (batch, count),
        torch.finfo(points.dtype).max,
        dtype=points.dtype,
        device=points.device,
    )
    farthest = torch.zeros(batch, dtype=torch.long, device=points.device)
    batch_indices = torch.arange(batch, device=points.device)
    for offset in range(sample_count):
        indices[:, offset] = farthest
        centroid = points[batch_indices, farthest][:, None]
        squared_distance = (points - centroid).square().sum(dim=-1)
        minimum_squared_distance = torch.minimum(
            minimum_squared_distance, squared_distance
        )
        farthest = minimum_squared_distance.argmax(dim=-1)
    return indices


def gather_operation(features: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather ``[B,C,N]`` features at ``[B,S]`` point indices."""

    if features.ndim != 3 or indices.ndim != 2:
        raise ValueError("gather expects features [B,C,N] and indices [B,S]")
    if features.shape[0] != indices.shape[0]:
        raise ValueError("gather batch dimensions do not align")
    if fps_backend() == LEGACY_FPS_BACKEND:
        return _legacy_gather_operation(features, indices)
    expanded = indices.long()[:, None].expand(-1, features.shape[1], -1)
    return torch.gather(features, 2, expanded)


def grouping_operation(features: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather ``[B,C,N]`` features into ``[B,C,S,K]`` neighborhoods."""

    if features.ndim != 3 or indices.ndim != 3:
        raise ValueError("grouping expects features [B,C,N] and indices [B,S,K]")
    if features.shape[0] != indices.shape[0]:
        raise ValueError("grouping batch dimensions do not align")
    if fps_backend() == LEGACY_FPS_BACKEND:
        return _legacy_grouping_operation(features, indices)
    batch = torch.arange(features.shape[0], device=features.device)[:, None, None]
    grouped = features.transpose(1, 2)[batch, indices.long()]
    return grouped.permute(0, 3, 1, 2).contiguous()


@torch.no_grad()
def ball_query(
    radius: float,
    sample_count: int,
    points: torch.Tensor,
    centers: torch.Tensor,
) -> torch.Tensor:
    """Return input-order radius neighborhoods for every query center."""

    if points.ndim != 3 or centers.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("ball query expects points/centers with shape [B,N,3]")
    if points.shape[0] != centers.shape[0] or centers.shape[-1] != 3:
        raise ValueError("ball-query point sets do not align")
    if radius <= 0.0 or sample_count <= 0:
        raise ValueError("ball-query radius and sample_count must be positive")
    count = points.shape[1]
    if count == 0:
        raise ValueError("cannot query an empty point cloud")
    if fps_backend() == LEGACY_FPS_BACKEND:
        return _legacy_ball_query(radius, sample_count, points, centers)
    retained = min(int(sample_count), count)
    squared_distance = (centers[:, :, None] - points[:, None]).square().sum(dim=-1)
    source_indices = torch.arange(count, device=points.device).view(1, 1, count)
    sentinel = torch.full_like(source_indices, count)
    candidates = torch.where(
        squared_distance < float(radius) ** 2,
        source_indices,
        sentinel,
    )
    selected = torch.topk(
        candidates, k=retained, dim=-1, largest=False, sorted=True
    ).values
    first = selected[..., :1]
    first = torch.where(first < count, first, torch.zeros_like(first))
    selected = torch.where(selected < count, selected, first)
    if retained < int(sample_count):
        selected = torch.cat(
            (selected, first.expand(-1, -1, int(sample_count) - retained)), dim=-1
        )
    return selected.long()
