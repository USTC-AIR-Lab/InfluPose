"""Stage2: observability-guided pose information field.

The network consumes only a partial camera-frame point cloud and geometric
observation features.  It predicts one canonical correspondence, one inlier
probability, an anisotropic precision and four pose-information channels for
each visible anchor. A weighted-Kabsch initialization followed by three
deterministic, differentiable anisotropic IRLS updates produces the only pose
and metric-box endpoint.

Category ids are never embedded or passed through a learned layer.  The
optional continuous-symmetry mask is used only after solving to choose the
same quotient representative as the REAL275 metric.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from shared import point_ops
from shared.pose_geometry import (
    axis_angle_to_matrix,
    canonicalize_continuous_y_rotation,
    matrix_to_pose6d,
)


def _gather(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if values.ndim != 3 or indices.ndim != 2:
        raise ValueError("gather expects values [B,N,C] and indices [B,K]")
    return torch.gather(
        values, 1, indices[..., None].expand(-1, -1, values.shape[-1])
    )


def _skew(values: torch.Tensor) -> torch.Tensor:
    """Return matrices satisfying ``skew(v) @ w == v cross w``."""

    x, y, z = values.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    return torch.stack(
        (zeros, -z, y, z, zeros, -x, -y, x, zeros), dim=-1
    ).reshape(*values.shape[:-1], 3, 3)


def weighted_kabsch(
    canonical_metric: torch.Tensor,
    camera_xyz: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the stable weighted Procrustes pose and covariance spectrum."""

    if canonical_metric.shape != camera_xyz.shape or canonical_metric.shape[-1] != 3:
        raise ValueError("Kabsch point sets must align as [B,K,3]")
    if weights.shape != canonical_metric.shape[:2]:
        raise ValueError("Kabsch weights must align as [B,K]")
    with torch.autocast(device_type=canonical_metric.device.type, enabled=False):
        source = canonical_metric.float()
        target = camera_xyz.float()
        numeric = weights.float().clamp_min(1e-5)
        numeric = numeric / numeric.sum(dim=1, keepdim=True).clamp_min(1e-6)
        source_center = (numeric[..., None] * source).sum(dim=1)
        target_center = (numeric[..., None] * target).sum(dim=1)
        source_zero = source - source_center[:, None]
        target_zero = target - target_center[:, None]
        covariance = torch.einsum(
            "bk,bki,bkj->bij", numeric, source_zero, target_zero
        )
        u, singular, vh = torch.linalg.svd(covariance, full_matrices=False)
        v = vh.transpose(-1, -2)
        candidate = v @ u.transpose(-1, -2)
        sign = torch.where(
            torch.linalg.det(candidate) < 0.0,
            -torch.ones_like(singular[:, 0]),
            torch.ones_like(singular[:, 0]),
        )
        correction = torch.eye(3, device=source.device, dtype=source.dtype)[None].repeat(
            source.shape[0], 1, 1
        )
        correction[:, 2, 2] = sign
        rotation = v @ correction @ u.transpose(-1, -2)
        translation = target_center - torch.einsum(
            "bij,bj->bi", rotation, source_center
        )
    return rotation, translation, singular


def information_irls(
    canonical: torch.Tensor,
    camera_xyz: torch.Tensor,
    initial_size: torch.Tensor,
    inlier_probability: torch.Tensor,
    axis_precision: torch.Tensor,
    information: torch.Tensor,
    *,
    initial_information: torch.Tensor | None = None,
    iterations: int,
    damping: float,
    max_rotation_step_degrees: float,
    max_translation_step: float,
    max_log_size_step: float,
) -> dict[str, torch.Tensor]:
    """Jointly solve rotation, translation and size with block-aware IRLS.

    The four information channels are ``up, yaw, translation, size``. They
    gate the corresponding Gauss--Newton columns per visible anchor. Residual
    axes are independently whitened by the learned anisotropic precision and
    a Cauchy weight suppresses inconsistent correspondences at every update.
    """

    if information.shape != (*canonical.shape[:2], 4):
        raise ValueError("information must have shape [B,K,4]")
    if initial_information is None:
        initial_information = information
    if initial_information.shape != information.shape:
        raise ValueError("initial_information must align with information")
    if int(iterations) != iterations or int(iterations) < 0:
        raise ValueError("IRLS iterations must be a nonnegative integer")
    positive_solver_parameters = {
        "IRLS damping": damping,
        "maximum rotation step": max_rotation_step_degrees,
        "maximum translation step": max_translation_step,
        "maximum log-size step": max_log_size_step,
    }
    for name, value in positive_solver_parameters.items():
        numeric = float(value)
        if not math.isfinite(numeric) or numeric <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    batch, count, _ = canonical.shape
    scalar_precision = axis_precision.mean(dim=-1)
    initial_weight = (
        inlier_probability
        * scalar_precision
        * initial_information[..., :3].mean(dim=-1)
    ).clamp_min(1e-5)
    rotation, translation, singular = weighted_kabsch(
        canonical * initial_size[:, None], camera_xyz, initial_weight
    )
    initial_rotation = rotation
    initial_translation = translation
    log_size = torch.log(initial_size.clamp_min(1e-4))
    identity9 = torch.eye(9, device=canonical.device, dtype=torch.float32)[None]
    gates: list[torch.Tensor] = []
    conditions: list[torch.Tensor] = []
    residual_rms: list[torch.Tensor] = []

    with torch.autocast(device_type=canonical.device.type, enabled=False):
        canonical_f = canonical.float()
        camera_f = camera_xyz.float()
        inlier_f = inlier_probability.float().clamp(1e-4, 1.0)
        precision_f = axis_precision.float().clamp(0.05, 50.0)
        information_f = information.float().clamp(0.02, 8.0)
        rotation = rotation.float()
        translation = translation.float()
        log_size = log_size.float()
        for _ in range(max(0, int(iterations))):
            size = log_size.exp().clamp(0.01, 0.80)
            canonical_metric = canonical_f * size[:, None]
            rotated = torch.einsum("bij,bkj->bki", rotation, canonical_metric)
            prediction = rotated + translation[:, None]
            residual = prediction - camera_f
            standardized_norm = torch.sqrt(
                (residual.square() * precision_f).sum(dim=-1).clamp_min(1e-12)
            )
            cauchy = 1.0 / (1.0 + standardized_norm.square())
            row_weight = inlier_f * cauchy

            rotation_jacobian = -_skew(rotated)
            translation_jacobian = torch.eye(
                3, device=canonical.device, dtype=torch.float32
            )[None, None].expand(batch, count, -1, -1)
            size_jacobian = rotation[:, None] * canonical_metric[:, :, None, :]
            jacobian = torch.cat(
                (rotation_jacobian, translation_jacobian, size_jacobian), dim=-1
            )

            object_up = F.normalize(rotation[:, :, 1], dim=-1, eps=1e-6)
            yaw_alignment = object_up.square()[:, None]
            up_info = information_f[..., 0:1]
            yaw_info = information_f[..., 1:2]
            rotation_info = (
                up_info * (1.0 - yaw_alignment) + yaw_info * yaw_alignment
            )
            parameter_info = torch.cat(
                (
                    rotation_info,
                    information_f[..., 2:3].expand(-1, -1, 3),
                    information_f[..., 3:4].expand(-1, -1, 3),
                ),
                dim=-1,
            ).clamp(0.02, 8.0)
            effective_jacobian = jacobian * parameter_info.sqrt()[:, :, None]
            weighted_jacobian = effective_jacobian * (
                row_weight[..., None, None] * precision_f[..., :, None]
            )
            normal = torch.einsum(
                "bkri,bkrj->bij", effective_jacobian, weighted_jacobian
            )
            gradient = torch.einsum(
                "bkri,bkr,bkr->bi",
                effective_jacobian,
                precision_f,
                row_weight[..., None] * residual,
            )
            eigenvalues = torch.linalg.eigvalsh(normal).clamp_min(1e-9)
            condition = eigenvalues[:, -1] / eigenvalues[:, 0]
            confidence = inlier_f.mean(dim=1) * information_f.mean(dim=(1, 2))
            gate = torch.sigmoid((confidence - 0.20) * 8.0)
            gate = gate * torch.sigmoid(
                (7.0 - torch.log10(condition.clamp_min(1.0))) * 2.0
            )
            diagonal_mean = normal.diagonal(dim1=-2, dim2=-1).mean(dim=-1)
            damped = normal + (
                damping + 1e-4 * diagonal_mean
            )[:, None, None] * identity9
            delta = torch.linalg.solve(damped, -gradient[..., None]).squeeze(-1)
            delta = torch.nan_to_num(delta, 0.0)

            rotation_delta = delta[:, :3]
            rotation_limit = math.radians(max_rotation_step_degrees)
            rotation_norm = torch.linalg.vector_norm(
                rotation_delta, dim=-1, keepdim=True
            )
            rotation_delta = rotation_delta * (
                rotation_limit / rotation_norm.clamp_min(rotation_limit)
            )
            translation_delta = delta[:, 3:6]
            translation_norm = torch.linalg.vector_norm(
                translation_delta, dim=-1, keepdim=True
            )
            translation_delta = translation_delta * (
                max_translation_step
                / translation_norm.clamp_min(max_translation_step)
            )
            size_delta = delta[:, 6:9].clamp(
                -max_log_size_step, max_log_size_step
            )
            rotation = axis_angle_to_matrix(gate[:, None] * rotation_delta) @ rotation
            translation = translation + gate[:, None] * translation_delta
            log_size = log_size + gate[:, None] * size_delta
            gates.append(gate)
            conditions.append(condition)
            residual_rms.append(residual.square().mean(dim=(1, 2)).sqrt())

    size = log_size.exp().clamp(0.01, 0.80)
    return {
        "rotation": rotation,
        "translation": translation,
        "size": size,
        "initial_rotation": initial_rotation,
        "initial_translation": initial_translation,
        "initial_size": initial_size,
        "initial_singular_values": singular,
        "solver_gate": (
            torch.stack(gates, dim=1)
            if gates
            else canonical.new_ones(batch, 0)
        ),
        "solver_condition": (
            torch.stack(conditions, dim=1)
            if conditions
            else canonical.new_ones(batch, 0)
        ),
        "solver_residual_rms": (
            torch.stack(residual_rms, dim=1)
            if residual_rms
            else canonical.new_zeros(batch, 0)
        ),
    }


class CameraRelationBlock(nn.Module):
    """Self-attention with rigid-invariant geometric relation biases."""

    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        if width <= 0 or heads <= 0 or width % heads:
            raise ValueError("relation width must be divisible by heads")
        self.width = int(width)
        self.heads = int(heads)
        self.head_dim = self.width // self.heads
        self.norm = nn.LayerNorm(self.width)
        self.qkv = nn.Linear(self.width, 3 * self.width, bias=False)
        self.relation_bias = nn.Sequential(
            nn.Linear(4, self.width),
            nn.SiLU(),
            nn.Linear(self.width, self.heads),
        )
        self.projection = nn.Linear(self.width, self.width)
        self.dropout = nn.Dropout(dropout)
        self.ff_norm = nn.LayerNorm(self.width)
        self.ff = nn.Sequential(
            nn.Linear(self.width, 4 * self.width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * self.width, self.width),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        xyz: torch.Tensor,
        normals: torch.Tensor,
        rays: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        batch, count, width = tokens.shape
        if width != self.width or xyz.shape != (batch, count, 3):
            raise ValueError("relation token/xyz contract drifted")
        if normals.shape != xyz.shape or rays.shape != xyz.shape:
            raise ValueError("relation directions must align with xyz")
        normalized = self.norm(tokens)
        qkv = self.qkv(normalized).reshape(
            batch, count, 3, self.heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=2)
        logits = torch.einsum("bihd,bjhd->bhij", query, key)
        logits = logits * (self.head_dim ** -0.5)
        distance = torch.cdist(xyz, xyz) / scale[:, None]
        normal_cosine = torch.einsum("bid,bjd->bij", normals, normals).clamp(-1, 1)
        ray_cosine = torch.einsum("bid,bjd->bij", rays, rays).clamp(-1, 1)
        depth_delta = (
            xyz[:, :, 2, None] - xyz[:, None, :, 2]
        ).abs() / scale[:, None]
        relation = torch.stack(
            (distance, normal_cosine, ray_cosine, depth_delta), dim=-1
        )
        bias = self.relation_bias(relation).permute(0, 3, 1, 2)
        attention = F.softmax(logits + bias, dim=-1)
        attended = torch.einsum("bhij,bjhd->bihd", attention, value).reshape(
            batch, count, width
        )
        tokens = tokens + self.dropout(self.projection(attended))
        return tokens + self.ff(self.ff_norm(tokens))


class Stage2PoseInformationField(nn.Module):
    """Partial point cloud -> per-point sufficient information -> one pose."""

    model_version = "stage2_observability_guided_pose_information_field_irls_v1"
    constructor_fields = (
        "observation_feature_dim",
        "token_width",
        "anchor_count",
        "relation_layers",
        "heads",
        "dropout",
        "canonical_range",
        "size_min",
        "size_max",
        "size_init",
        "irls_iterations",
        "irls_damping",
        "max_rotation_step_degrees",
        "max_translation_step",
        "max_log_size_step",
    )

    def __init__(
        self,
        *,
        observation_feature_dim: int = 14,
        token_width: int = 192,
        anchor_count: int = 192,
        relation_layers: int = 4,
        heads: int = 6,
        dropout: float = 0.05,
        canonical_range: float = 0.65,
        size_min: float = 0.01,
        size_max: float = 0.80,
        size_init: float = 0.20,
        irls_iterations: int = 3,
        irls_damping: float = 1e-3,
        max_rotation_step_degrees: float = 8.0,
        max_translation_step: float = 0.015,
        max_log_size_step: float = 0.12,
    ) -> None:
        super().__init__()
        if observation_feature_dim != 14:
            raise ValueError("Stage2 currently requires the 14-D observation contract")
        if min(token_width, anchor_count, heads, canonical_range) <= 0:
            raise ValueError("Stage2 dimensions must be positive")
        if relation_layers <= 0:
            raise ValueError("relation_layers must be positive")
        if token_width % heads:
            raise ValueError("token_width must be divisible by heads")
        if not 0.0 < size_min < size_init < size_max:
            raise ValueError("size bounds must contain size_init")
        if int(irls_iterations) != irls_iterations or int(irls_iterations) < 0:
            raise ValueError("irls_iterations must be a nonnegative integer")
        positive_solver_parameters = {
            "irls_damping": irls_damping,
            "max_rotation_step_degrees": max_rotation_step_degrees,
            "max_translation_step": max_translation_step,
            "max_log_size_step": max_log_size_step,
        }
        for name, value in positive_solver_parameters.items():
            numeric = float(value)
            if not math.isfinite(numeric) or numeric <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        self.observation_feature_dim = int(observation_feature_dim)
        self.token_width = int(token_width)
        self.anchor_count = int(anchor_count)
        self.relation_layers = int(relation_layers)
        self.heads = int(heads)
        self.dropout = float(dropout)
        self.use_view_evidence = True
        self.information_channels = 4
        self.canonical_range = float(canonical_range)
        self.size_min = float(size_min)
        self.size_max = float(size_max)
        self.size_init = float(size_init)
        self.irls_iterations = int(irls_iterations)
        self.irls_damping = float(irls_damping)
        self.max_rotation_step_degrees = float(max_rotation_step_degrees)
        self.max_translation_step = float(max_translation_step)
        self.max_log_size_step = float(max_log_size_step)

        input_dim = 3 + self.observation_feature_dim
        self.point_stem = nn.Sequential(
            nn.Linear(input_dim, self.token_width),
            nn.LayerNorm(self.token_width),
            nn.SiLU(),
            nn.Linear(self.token_width, self.token_width),
            nn.LayerNorm(self.token_width),
            nn.SiLU(),
        )
        self.relations = nn.ModuleList(
            CameraRelationBlock(self.token_width, self.heads, self.dropout)
            for _ in range(self.relation_layers)
        )
        self.global_projection = nn.Sequential(
            nn.Linear(2 * self.token_width + 7, self.token_width),
            nn.LayerNorm(self.token_width),
            nn.SiLU(),
            nn.Linear(self.token_width, self.token_width),
            nn.LayerNorm(self.token_width),
            nn.SiLU(),
        )
        self.token_fusion = nn.Sequential(
            nn.Linear(2 * self.token_width, self.token_width),
            nn.LayerNorm(self.token_width),
            nn.SiLU(),
        )
        self.canonical_head = nn.Linear(self.token_width, 3)
        self.inlier_head = nn.Linear(self.token_width, 1)
        self.precision_head = nn.Linear(self.token_width, 3)
        self.information_head = nn.Linear(self.token_width, 4)
        self.size_head = nn.Linear(self.token_width, 3)
        self._reset_outputs()

    def _reset_outputs(self) -> None:
        nn.init.normal_(self.canonical_head.weight, std=0.01)
        nn.init.zeros_(self.canonical_head.bias)
        nn.init.zeros_(self.inlier_head.weight)
        nn.init.constant_(self.inlier_head.bias, 2.0)
        nn.init.zeros_(self.precision_head.weight)
        nn.init.constant_(self.precision_head.bias, 1.0)
        nn.init.zeros_(self.information_head.weight)
        nn.init.constant_(self.information_head.bias, 0.5)
        nn.init.zeros_(self.size_head.weight)
        fraction = (self.size_init - self.size_min) / (self.size_max - self.size_min)
        nn.init.constant_(self.size_head.bias, math.log(fraction / (1.0 - fraction)))

    def architecture(self) -> dict[str, int | float]:
        return {
            "observation_feature_dim": self.observation_feature_dim,
            "token_width": self.token_width,
            "anchor_count": self.anchor_count,
            "relation_layers": self.relation_layers,
            "heads": self.heads,
            "dropout": self.dropout,
            "canonical_range": self.canonical_range,
            "size_min": self.size_min,
            "size_max": self.size_max,
            "size_init": self.size_init,
            "irls_iterations": self.irls_iterations,
            "irls_damping": self.irls_damping,
            "max_rotation_step_degrees": self.max_rotation_step_degrees,
            "max_translation_step": self.max_translation_step,
            "max_log_size_step": self.max_log_size_step,
        }

    @classmethod
    def from_architecture(cls, architecture: dict[str, object]) -> Stage2PoseInformationField:
        """Build only from the exact G10F/IRLS Stage2 architecture contract."""
        if not isinstance(architecture, dict):
            raise TypeError("checkpoint architecture must be a dictionary")
        unknown = sorted(set(architecture) - set(cls.constructor_fields))
        if unknown:
            raise ValueError(f"unsupported Stage2 architecture fields: {unknown}")
        return cls(**architecture)

    def _prepare_observation_features(
        self, observation_features: torch.Tensor
    ) -> torch.Tensor:
        return observation_features

    def _predict_information(
        self, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.information_head(tokens)
        return logits, F.softplus(logits) + 0.02

    def _solver_weights(
        self,
        inlier_probability: torch.Tensor,
        axis_precision: torch.Tensor,
        information: torch.Tensor,
    ) -> torch.Tensor:
        scalar_precision = axis_precision.mean(dim=-1)
        return (
            inlier_probability
            * scalar_precision
            * information[..., :3].mean(dim=-1)
        ).clamp_min(1e-5)

    def forward(
        self,
        points: torch.Tensor,
        normals: torch.Tensor,
        observation_features: torch.Tensor,
        *,
        alignment_center: torch.Tensor,
        continuous_symmetry: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError("points must have shape [B,N,3]")
        if normals.shape != points.shape:
            raise ValueError("normals must align with points")
        if observation_features.shape != (*points.shape[:2], self.observation_feature_dim):
            raise ValueError("observation features do not align with points")
        if alignment_center.shape != (points.shape[0], 3):
            raise ValueError("alignment_center must have shape [B,3]")
        if continuous_symmetry is not None and continuous_symmetry.shape != points.shape[:1]:
            raise ValueError("continuous_symmetry must have shape [B]")

        relative = points - alignment_center[:, None]
        extent = relative.amax(dim=1) - relative.amin(dim=1)
        point_scale = extent.amax(dim=-1, keepdim=True).clamp_min(0.03)
        normalized_xyz = relative / point_scale[:, None]
        count = min(self.anchor_count, points.shape[1])
        indices = point_ops.farthest_point_sample(
            normalized_xyz.contiguous(), count
        ).long()
        anchor_xyz = _gather(points, indices)
        anchor_normals = F.normalize(_gather(normals, indices), dim=-1, eps=1e-6)
        anchor_observation = _gather(observation_features, indices)
        anchor_observation = self._prepare_observation_features(anchor_observation)
        anchor_relative = _gather(normalized_xyz, indices)
        rays = F.normalize(anchor_xyz, dim=-1, eps=1e-6)
        tokens = self.point_stem(torch.cat((anchor_relative, anchor_observation), dim=-1))
        for block in self.relations:
            tokens = block(tokens, anchor_xyz, anchor_normals, rays, point_scale)

        statistics = torch.cat(
            (
                alignment_center,
                point_scale,
                extent / point_scale,
            ),
            dim=-1,
        )
        pooled = torch.cat((tokens.amax(dim=1), tokens.mean(dim=1), statistics), dim=-1)
        global_token = self.global_projection(pooled)
        tokens = self.token_fusion(
            torch.cat((tokens, global_token[:, None].expand_as(tokens)), dim=-1)
        )
        canonical = torch.tanh(self.canonical_head(tokens)) * self.canonical_range
        inlier_logits = self.inlier_head(tokens).squeeze(-1)
        axis_precision = F.softplus(self.precision_head(tokens)) + 0.05
        information_logits, information = self._predict_information(tokens)
        raw_size = self.size_head(global_token)
        initial_size = (
            self.size_min
            + (self.size_max - self.size_min) * torch.sigmoid(raw_size)
        )

        solver_weights = self._solver_weights(
            torch.sigmoid(inlier_logits), axis_precision, information
        )
        solved = information_irls(
            canonical,
            anchor_xyz,
            initial_size,
            torch.sigmoid(inlier_logits),
            axis_precision,
            information,
            iterations=self.irls_iterations,
            damping=self.irls_damping,
            max_rotation_step_degrees=self.max_rotation_step_degrees,
            max_translation_step=self.max_translation_step,
            max_log_size_step=self.max_log_size_step,
        )
        rotation = solved["rotation"]
        translation = solved["translation"]
        metric_size = solved["size"]
        if continuous_symmetry is not None and torch.any(continuous_symmetry):
            dummy_ids = torch.where(
                continuous_symmetry,
                torch.zeros_like(continuous_symmetry, dtype=torch.long),
                torch.full_like(continuous_symmetry, 2, dtype=torch.long),
            )
            rotation = canonicalize_continuous_y_rotation(rotation, dummy_ids)
        pose = torch.cat((matrix_to_pose6d(rotation), translation), dim=-1)
        return {
            "pose": pose,
            "size": metric_size,
            "rotation": rotation,
            "translation": translation,
            "initial_rotation": solved["initial_rotation"],
            "initial_translation": solved["initial_translation"],
            "network_initial_size": initial_size,
            "anchor_indices": indices,
            "anchor_xyz": anchor_xyz,
            "anchor_normals": anchor_normals,
            "canonical": canonical,
            "inlier_logits": inlier_logits,
            "axis_precision": axis_precision,
            "information_logits": information_logits,
            "information": information,
            "point_scale": point_scale,
            "solver_weights": solver_weights,
            "solver_singular_values": solved["initial_singular_values"],
            "initial_singular_values": solved["initial_singular_values"],
            "solver_gate": solved["solver_gate"],
            "solver_condition": solved["solver_condition"],
            "solver_residual_rms": solved["solver_residual_rms"],
        }
