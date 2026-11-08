"""Auxiliary objectives for Stage2 refinement."""

from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F

from shared.point_features import estimate_camera_observation_features
from shared.pose_geometry import pose6d_to_matrix, quotient_rotation_distance
from shared.training import _class_balanced_mean, pose_box_loss, symmetry_aware_box_iou_proxy


AUXILIARY_DEFAULTS = {
    "view_consistency_weight": 0.5,
    "view_fraction": 0.25,
    "view_interval": 4,
    "view_dropout_max": 0.25,
    "view_noise_std": 0.0025,
    "view_axial_noise_std": 0.004,
    "reference_pose_weight": 2.0,
    "reference_iou_weight": 2.0,
    "reference_rotation_margin_degrees": 0.25,
    "reference_translation_margin_cm": 0.10,
    "reference_iou_margin": 0.01,
}


def validate_auxiliary_config(cfg) -> None:
    for name in AUXILIARY_DEFAULTS:
        value = getattr(cfg, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if not 0 <= cfg.view_fraction <= 1 or not 0 <= cfg.view_dropout_max < 1:
        raise ValueError("view fraction or dropout is invalid")
    if cfg.view_interval < 1:
        raise ValueError("view_interval must be positive")
    if cfg.view_consistency_weight > 0 and cfg.view_fraction == 0:
        raise ValueError("view consistency requires a positive view_fraction")


def reference_enabled(cfg) -> bool:
    return cfg.reference_pose_weight > 0 or cfg.reference_iou_weight > 0


def frozen_reference(model, state_dict=None):
    """Create a gradient-free evaluation copy for the training reference."""
    reference = copy.deepcopy(model)
    if state_dict is not None:
        reference.load_state_dict(state_dict, strict=True)
    reference.eval()
    reference.requires_grad_(False)
    return reference


def detached_state_dict(model):
    """Return independent CPU tensors, including when the model is on CPU."""
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def view_consistency_loss(prediction, primary, class_ids, handles, size_loss_weight):
    """Match pose and size across views with a stop-gradient primary target."""
    loss, _ = pose_box_loss(
        prediction, primary["pose"].detach(), primary["size"].detach(),
        class_ids, handles, size_loss_weight,
    )
    return loss


def forward_training_views(
    model, points, normals, observation, center, symmetric, class_ids, handles,
    cfg, *, batch_index, enabled=True,
):
    """Run primary and perturbed views in one forward, including under DDP.

    The second observation retains the primary camera frame and alignment
    center. Its target is the detached primary pose and size.
    """
    count = points.shape[0]
    use_view = (
        enabled
        and (cfg.view_consistency_weight > 0 or cfg.refine_view_pose_weight > 0)
        and batch_index % cfg.view_interval == 0
    )
    view_count = max(1, round(count * cfg.view_fraction)) if use_view else 0
    if use_view:
        indices = torch.randperm(count, device=points.device)[:view_count]
        second_points = robust_partial_view(
            points[indices], dropout_max=cfg.view_dropout_max,
            noise_std=cfg.view_noise_std, axial_noise_std=cfg.view_axial_noise_std,
        )
        with torch.autocast(device_type=points.device.type, enabled=False):
            second_normals, second_observation = estimate_camera_observation_features(
                second_points.float(), center[indices].float(), k=cfg.normals_k,
            )
        inputs = (
            torch.cat((points, second_points)),
            torch.cat((normals, second_normals.detach())),
            torch.cat((observation, second_observation.detach())),
        )
        all_centers = torch.cat((center, center[indices]))
        all_symmetric = torch.cat((symmetric, symmetric[indices]))
    else:
        inputs = (points, normals, observation)
        all_centers, all_symmetric = center, symmetric
    combined = model(
        *inputs, alignment_center=all_centers, continuous_symmetry=all_symmetric,
    )
    primary = {key: value[:count] for key, value in combined.items()}
    view_loss = points.new_zeros(())
    view_data = None
    if use_view:
        secondary = {key: value[count:] for key, value in combined.items()}
        target = {key: value[indices] for key, value in primary.items()}
        view_loss = view_consistency_loss(
            secondary, target, class_ids[indices], handles[indices],
            cfg.size_loss_weight,
        )
        view_data = (secondary, indices)
    return primary, view_loss, view_count, view_data


def robust_partial_view(
    points: torch.Tensor,
    *,
    dropout_max: float,
    noise_std: float,
    axial_noise_std: float,
) -> torch.Tensor:
    """Create a fixed-size detector-style second view without changing its frame."""

    if not 0.0 <= dropout_max < 1.0 or min(noise_std, axial_noise_std) < 0.0:
        raise ValueError("robust partial-view corruption is invalid")
    batch, count, _ = points.shape
    dropout_probability = torch.rand(
        batch, 1, device=points.device, dtype=points.dtype
    ) * dropout_max
    dropped = torch.rand(batch, count, device=points.device) < dropout_probability
    replacement_indices = torch.randint(
        count, (batch, count), device=points.device
    )
    replacements = torch.gather(
        points, 1, replacement_indices[..., None].expand(-1, -1, 3)
    )
    corrupted = torch.where(dropped[..., None], replacements, points)
    if noise_std:
        corrupted = corrupted + torch.randn_like(corrupted) * noise_std
    if axial_noise_std:
        rays = F.normalize(corrupted, dim=-1, eps=1e-6)
        axial = torch.randn(
            batch, count, 1, device=points.device, dtype=points.dtype
        ) * axial_noise_std
        corrupted = corrupted + rays * axial
    return corrupted


def reference_non_regression_loss(
    prediction: dict[str, torch.Tensor],
    reference_prediction: dict[str, torch.Tensor],
    target_pose: torch.Tensor,
    target_size: torch.Tensor,
    class_ids: torch.Tensor,
    handles: torch.Tensor,
    *,
    student_iou: torch.Tensor | None = None,
    rotation_margin_degrees: float,
    translation_margin_cm: float,
    iou_margin: float,
    class_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Penalize degradation beyond a tolerance relative to a frozen predictor.

    Both predictions are compared with the training targets. Improvements
    over the reference incur no penalty. The reference has no gradients.
    """

    if min(rotation_margin_degrees, translation_margin_cm, iou_margin) < 0.0:
        raise ValueError("reference non-regression margins must be non-negative")
    target_rotation = pose6d_to_matrix(target_pose[:, :6])
    student_rotation_error = quotient_rotation_distance(
        pose6d_to_matrix(prediction["pose"][:, :6]),
        target_rotation,
        class_ids,
        handles,
    )
    with torch.no_grad():
        reference_rotation_error = quotient_rotation_distance(
            pose6d_to_matrix(reference_prediction["pose"][:, :6]),
            target_rotation,
            class_ids,
            handles,
        )
    student_translation_error = torch.linalg.vector_norm(
        prediction["pose"][:, 6:] - target_pose[:, 6:], dim=-1
    )
    reference_translation_error = torch.linalg.vector_norm(
        reference_prediction["pose"][:, 6:].detach() - target_pose[:, 6:], dim=-1
    )
    rotation_regret = F.relu(
        student_rotation_error
        - reference_rotation_error
        - math.radians(rotation_margin_degrees)
    )
    translation_regret = F.relu(
        student_translation_error
        - reference_translation_error
        - translation_margin_cm / 100.0
    )
    pose_regret = _class_balanced_mean(
        rotation_regret + 2.0 * translation_regret,
        class_ids,
        class_weights=class_weights,
    )

    if student_iou is None:
        student_iou = symmetry_aware_box_iou_proxy(
            prediction["pose"],
            prediction["size"],
            target_pose,
            target_size,
            class_ids,
            handles,
        )
    with torch.no_grad():
        reference_iou = symmetry_aware_box_iou_proxy(
            reference_prediction["pose"],
            reference_prediction["size"],
            target_pose,
            target_size,
            class_ids,
            handles,
        )
    iou_regret = _class_balanced_mean(
        F.relu(reference_iou - student_iou - iou_margin),
        class_ids,
        class_weights=class_weights,
    )
    return pose_regret, iou_regret, {
        "reference_rotation_regression_deg": (
            rotation_regret.mean().detach() * (180.0 / math.pi)
        ),
        "reference_translation_regression_cm": (
            translation_regret.mean().detach() * 100.0
        ),
        "reference_iou_regression": iou_regret.detach(),
    }
