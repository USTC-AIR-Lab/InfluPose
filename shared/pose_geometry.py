"""Differentiable 6D-pose geometry and REAL275 symmetry handling."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from pytorch3d.transforms import (
    axis_angle_to_matrix as _pytorch3d_axis_angle_to_matrix,
    matrix_to_rotation_6d as _pytorch3d_matrix_to_rotation_6d,
    rotation_6d_to_matrix as _pytorch3d_rotation_6d_to_matrix,
    so3_log_map as _pytorch3d_so3_log_map,
)


AXIS_SYMMETRIC_CLASS_IDS = (0, 1, 3)


def rotation_6d_to_matrix(rotation: torch.Tensor) -> torch.Tensor:
    """Use the exact PyTorch3D conversion used by the archived runs."""

    return _pytorch3d_rotation_6d_to_matrix(rotation)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """Use the exact PyTorch3D matrix-to-6D conversion."""

    return _pytorch3d_matrix_to_rotation_6d(matrix)


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Use the exact PyTorch3D axis-angle conversion used by G10A/G10F."""

    return _pytorch3d_axis_angle_to_matrix(axis_angle)


def so3_log_map(matrix: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Use the exact PyTorch3D SO(3) logarithm used by G10A/G10F."""

    return _pytorch3d_so3_log_map(matrix, eps=eps)


def pose6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    """Convert the project's transposed 6D pose convention to SO(3)."""

    return rotation_6d_to_matrix(rot6d).transpose(-1, -2)


def matrix_to_pose6d(matrix: torch.Tensor) -> torch.Tensor:
    """Convert SO(3) to the project's transposed 6D pose convention."""

    return matrix_to_rotation_6d(matrix.transpose(-1, -2))


def _continuous_y_mask(
    class_ids: torch.Tensor, handle_visibility: torch.Tensor | None = None
) -> torch.Tensor:
    mask = torch.zeros_like(class_ids, dtype=torch.bool)
    for class_id in AXIS_SYMMETRIC_CLASS_IDS:
        mask |= class_ids == class_id
    if handle_visibility is not None:
        mask |= (class_ids == 5) & (handle_visibility == 0)
    return mask


def continuous_y_symmetry_mask(
    class_ids: torch.Tensor, handle_visibility: torch.Tensor | None = None
) -> torch.Tensor:
    """Public form of the REAL275 continuous-y symmetry predicate."""

    return _continuous_y_mask(class_ids, handle_visibility)


def canonicalize_continuous_y_rotation(
    matrix: torch.Tensor,
    class_ids: torch.Tensor,
    handle_visibility: torch.Tensor | None = None,
) -> torch.Tensor:
    """Choose a prediction-independent representative for continuous-y classes.

    REAL275 ignores rotation around the object's local y axis for bottle, bowl,
    can, and mugs whose handle is hidden.  The deterministic model still needs
    one concrete representative.  This function fixes that gauge using only
    the quotient-invariant camera-frame y axis: camera x is projected onto its
    normal plane, with camera z as a stable fallback near collinearity.

    Unlike nearest-representative alignment, this target never depends on the
    current model prediction.  The returned matrix is in the same SO(3)/G
    equivalence class as the input and non-symmetric examples are unchanged.
    """

    if matrix.ndim != 3 or matrix.shape[-2:] != (3, 3):
        raise ValueError("matrix must have shape [N,3,3]")
    if class_ids.shape != matrix.shape[:1]:
        raise ValueError("class_ids must have shape [N]")
    if handle_visibility is not None and handle_visibility.shape != class_ids.shape:
        raise ValueError("handle_visibility must align with class_ids")

    symmetric = _continuous_y_mask(class_ids, handle_visibility)
    if not torch.any(symmetric):
        return matrix

    y_axis = F.normalize(matrix[:, :, 1], dim=-1)
    camera_x = matrix.new_tensor((1.0, 0.0, 0.0))[None].expand_as(y_axis)
    camera_z = matrix.new_tensor((0.0, 0.0, 1.0))[None].expand_as(y_axis)
    use_z = torch.abs((camera_x * y_axis).sum(dim=-1)) > 0.90
    reference = torch.where(use_z[:, None], camera_z, camera_x)
    x_axis = reference - (reference * y_axis).sum(dim=-1, keepdim=True) * y_axis
    x_axis = F.normalize(x_axis, dim=-1)
    z_axis = F.normalize(torch.linalg.cross(x_axis, y_axis, dim=-1), dim=-1)
    # Recompute x after the cross product so numerical drift cannot spoil
    # orthogonality or handedness.
    x_axis = F.normalize(torch.linalg.cross(y_axis, z_axis, dim=-1), dim=-1)
    canonical = torch.stack((x_axis, y_axis, z_axis), dim=-1)
    return torch.where(symmetric[:, None, None], canonical, matrix)


def quotient_rotation_distance(
    pred_matrix: torch.Tensor,
    target_matrix: torch.Tensor,
    class_ids: torch.Tensor,
    handle_visibility: torch.Tensor | None = None,
) -> torch.Tensor:
    """Geodesic SO(3)/G distance in radians.

    Leading sample dimensions are supported.  ``class_ids`` and
    ``handle_visibility`` are expected to contain the batch dimension only.
    """

    while class_ids.ndim < pred_matrix.ndim - 2:
        class_ids = class_ids.unsqueeze(-1)
    if handle_visibility is not None:
        while handle_visibility.ndim < pred_matrix.ndim - 2:
            handle_visibility = handle_visibility.unsqueeze(-1)

    relative = pred_matrix @ target_matrix.transpose(-1, -2)
    trace = relative.diagonal(dim1=-2, dim2=-1).sum(-1)
    full_cos = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    full_distance = torch.acos(full_cos)

    pred_y = F.normalize(pred_matrix[..., :, 1], dim=-1)
    target_y = F.normalize(target_matrix[..., :, 1], dim=-1)
    axis_cos = (pred_y * target_y).sum(-1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    axis_distance = torch.acos(axis_cos)

    symmetric = _continuous_y_mask(class_ids, handle_visibility)
    return torch.where(symmetric, axis_distance, full_distance)
