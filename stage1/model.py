"""Stage1 complete cloud -> 128-D RAE -> pose and metric box.

The model deliberately has no category embedding, category-specific pose head,
or category size table.  ``class_ids`` are accepted only for API compatibility
and are never read.  A dual-resolution point encoder preserves both stable
global geometry and fine camera-frame directional evidence; one shared decoder
maps the fully entangled latent to rotation, translation, and absolute size.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from shared import point_ops
from stage1.encoders import HighResolutionDirectionalTransformer


def _identity_rotation_6d(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Identity in the repository's transposed 6D rotation convention."""

    return torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=dtype)


def _mlp(dimensions: list[int], *, dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    for index, (input_dim, output_dim) in enumerate(
        zip(dimensions[:-1], dimensions[1:])
    ):
        layers.append(nn.Linear(input_dim, output_dim))
        if index + 1 < len(dimensions) - 1:
            layers.extend((nn.LayerNorm(output_dim), nn.SiLU()))
            if dropout:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class LocalGeometricAttentionStage(nn.Module):
    """FPS + ball-query point attention with camera-frame geometry preserved."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        sample_count: int,
        radius: float,
        neighbors: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if min(input_dim, output_dim, sample_count, neighbors, heads) <= 0:
            raise ValueError("local geometric attention dimensions must be positive")
        if radius <= 0.0 or output_dim % heads:
            raise ValueError("local attention radius/head contract is invalid")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.sample_count = int(sample_count)
        self.radius = float(radius)
        self.neighbors = int(neighbors)
        self.heads = int(heads)
        self.head_dim = self.output_dim // self.heads

        self.center_projection = nn.Linear(self.input_dim, self.output_dim)
        self.query = nn.Linear(self.output_dim, self.output_dim, bias=False)
        self.key = nn.Linear(self.input_dim, self.output_dim, bias=False)
        self.value = nn.Linear(self.input_dim, self.output_dim, bias=False)
        self.position = nn.Sequential(
            nn.Linear(3, self.output_dim),
            nn.LayerNorm(self.output_dim),
            nn.SiLU(),
            nn.Linear(self.output_dim, self.output_dim),
        )
        self.position_bias = nn.Linear(self.output_dim, self.heads, bias=False)
        self.output_projection = nn.Linear(self.output_dim, self.output_dim)
        self.output_norm = nn.LayerNorm(self.output_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(self.output_dim, 4 * self.output_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * self.output_dim, self.output_dim),
            nn.Dropout(dropout),
        )
        self.feed_forward_norm = nn.LayerNorm(self.output_dim)
        self.attention_dropout = nn.Dropout(dropout)

    @staticmethod
    def _gather(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return point_ops.gather_operation(
            values.transpose(1, 2).contiguous(), indices
        ).transpose(1, 2).contiguous()

    def forward(
        self, xyz: torch.Tensor, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if xyz.ndim != 3 or xyz.shape[-1] != 3:
            raise ValueError("local-attention xyz must have shape [B, N, 3]")
        if features.shape[:2] != xyz.shape[:2] or features.shape[-1] != self.input_dim:
            raise ValueError("local-attention features do not align with xyz")
        sample_count = min(self.sample_count, xyz.shape[1])
        sample_indices = point_ops.farthest_point_sample(
            xyz.contiguous(), sample_count
        )
        sampled_xyz = self._gather(xyz, sample_indices)
        sampled_features = self._gather(features, sample_indices)
        group_indices = point_ops.ball_query(
            self.radius,
            min(self.neighbors, xyz.shape[1]),
            xyz.contiguous(),
            sampled_xyz.contiguous(),
        )
        grouped_features = point_ops.grouping_operation(
            features.transpose(1, 2).contiguous(), group_indices
        ).permute(0, 2, 3, 1).contiguous()
        grouped_xyz = point_ops.grouping_operation(
            xyz.transpose(1, 2).contiguous(), group_indices
        ).permute(0, 2, 3, 1).contiguous()
        grouped_xyz = grouped_xyz - sampled_xyz[:, :, None]

        center = self.center_projection(sampled_features)
        query = self.query(center).reshape(
            center.shape[0], sample_count, self.heads, self.head_dim
        ).permute(0, 2, 1, 3)
        key = self.key(grouped_features).reshape(
            center.shape[0], sample_count, grouped_features.shape[2],
            self.heads, self.head_dim,
        ).permute(0, 3, 1, 2, 4)
        value = self.value(grouped_features).reshape(
            center.shape[0], sample_count, grouped_features.shape[2],
            self.heads, self.head_dim,
        ).permute(0, 3, 1, 2, 4)
        position = self.position(grouped_xyz)
        position_heads = position.reshape(
            center.shape[0], sample_count, grouped_features.shape[2],
            self.heads, self.head_dim,
        ).permute(0, 3, 1, 2, 4)
        logits = (
            query[:, :, :, None] * (key + position_heads)
        ).sum(dim=-1) / math.sqrt(float(self.head_dim))
        logits = logits + self.position_bias(position).permute(0, 3, 1, 2)
        attention = self.attention_dropout(torch.softmax(logits, dim=-1))
        aggregated = (attention[..., None] * (value + position_heads)).sum(dim=3)
        aggregated = aggregated.permute(0, 2, 1, 3).reshape(
            center.shape[0], sample_count, self.output_dim
        )
        features = self.output_norm(center + self.output_projection(aggregated))
        features = self.feed_forward_norm(features + self.feed_forward(features))
        return sampled_xyz, features


class CameraFrameMultiScaleGeometricTransformerEncoder(nn.Module):
    """Category-free local point transformer for complete camera-frame clouds.

    Three local-attention scales retain signed geometry, while a global
    transformer and explicit second/third-order camera-frame moments preserve
    the directional evidence needed for 6D rotation.
    """

    encoder_version = "camera_frame_multiscale_geometric_transformer_v2"

    def __init__(
        self,
        *,
        output_dim: int = 512,
        stage_dims: tuple[int, int, int] = (192, 256, 384),
        sample_counts: tuple[int, int, int] = (384, 192, 96),
        radii: tuple[float, float, float] = (0.12, 0.25, 0.45),
        neighbors: int = 32,
        local_heads: int = 8,
        transformer_layers: int = 4,
        transformer_heads: int = 8,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if len(stage_dims) != 3 or len(sample_counts) != 3 or len(radii) != 3:
            raise ValueError("geometric encoder requires exactly three scales")
        if min(output_dim, *stage_dims, *sample_counts, neighbors) <= 0:
            raise ValueError("geometric encoder dimensions must be positive")
        if any(dim % local_heads for dim in stage_dims):
            raise ValueError("each geometric stage dimension must divide local_heads")
        if stage_dims[-1] % transformer_heads:
            raise ValueError("final geometric stage must divide transformer_heads")
        self.point_feature_dim = 3
        self.output_dim = int(output_dim)
        self.class_dim = 0
        self.class_embedding = None
        self.stage_dims = tuple(int(value) for value in stage_dims)
        self.sample_counts = tuple(int(value) for value in sample_counts)
        self.radii = tuple(float(value) for value in radii)
        self.neighbors = int(neighbors)
        self.local_heads = int(local_heads)
        self.dropout = float(dropout)

        # xyz, normal, cross, dot, radius, xyz^2, xyz*r, normal*r = 20.
        self.input_embedding = nn.Sequential(
            nn.Linear(20, self.stage_dims[0]),
            nn.LayerNorm(self.stage_dims[0]),
            nn.SiLU(),
            nn.Linear(self.stage_dims[0], self.stage_dims[0]),
            nn.LayerNorm(self.stage_dims[0]),
        )
        input_dims = (self.stage_dims[0], self.stage_dims[0], self.stage_dims[1])
        self.local_stages = nn.ModuleList(
            LocalGeometricAttentionStage(
                input_dim,
                output_dim,
                sample_count=sample_count,
                radius=radius,
                neighbors=self.neighbors,
                heads=self.local_heads,
                dropout=dropout,
            )
            for input_dim, output_dim, sample_count, radius in zip(
                input_dims, self.stage_dims, self.sample_counts, self.radii
            )
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.stage_dims[-1],
            nhead=transformer_heads,
            dim_feedforward=4 * self.stage_dims[-1],
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.global_transformer = nn.TransformerEncoder(
            layer,
            num_layers=transformer_layers,
            norm=nn.LayerNorm(self.stage_dims[-1]),
        )
        self.statistics_dim = 60
        self.output_head = nn.Sequential(
            nn.Linear(2 * self.stage_dims[-1] + self.statistics_dim, 768),
            nn.LayerNorm(768),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(768, self.output_dim),
            nn.LayerNorm(self.output_dim),
        )

    def forward(
        self,
        points: torch.Tensor,
        class_ids: torch.Tensor | None,
        *,
        point_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError("points must have shape [batch, points, 3]")
        if point_features is None or point_features.shape != points.shape:
            raise ValueError("one camera-frame normal is required per point")
        if class_ids is not None and class_ids.shape != (points.shape[0],):
            raise ValueError("ignored class_ids must align with points")
        if not torch.isfinite(points).all() or not torch.isfinite(point_features).all():
            raise ValueError("geometric encoder inputs must be finite")

        center = points.mean(dim=1)
        relative = points - center[:, None]
        extent = relative.amax(dim=1) - relative.amin(dim=1)
        scale = extent.amax(dim=-1, keepdim=True).clamp_min(0.03)
        normalized = relative / scale[:, None]
        normals = torch.nn.functional.normalize(point_features, dim=-1, eps=1e-6)
        cross = torch.cross(normalized, normals, dim=-1)
        dot = (normalized * normals).sum(dim=-1, keepdim=True)
        radius = torch.linalg.vector_norm(normalized, dim=-1, keepdim=True)
        features = self.input_embedding(
            torch.cat(
                (
                    normalized,
                    normals,
                    cross,
                    dot,
                    radius,
                    normalized.square(),
                    normalized * radius,
                    normals * radius,
                ),
                dim=-1,
            )
        )
        xyz = normalized.contiguous()
        for stage in self.local_stages:
            xyz, features = stage(xyz, features)
        features = self.global_transformer(features)
        pooled = torch.cat((features.amax(dim=1), features.mean(dim=1)), dim=-1)

        covariance = torch.einsum(
            "bni,bnj->bij", normalized, normalized
        ) / max(1, normalized.shape[1])
        third = torch.einsum(
            "bni,bnj,bnk->bijk", normalized, normalized, normalized
        ) / max(1, normalized.shape[1])
        position_normal = torch.einsum(
            "bni,bnj->bij", normalized, normals
        ) / max(1, normalized.shape[1])
        radial = torch.cat(
            (
                radius.mean(dim=1),
                radius.std(dim=1),
                radius.amax(dim=1),
            ),
            dim=-1,
        )
        statistics = torch.cat(
            (
                center,
                extent,
                covariance.flatten(1),
                third.flatten(1),
                normals.mean(dim=1),
                normals.abs().mean(dim=1),
                position_normal.flatten(1),
                radial,
            ),
            dim=-1,
        )
        if statistics.shape[-1] != self.statistics_dim:
            raise RuntimeError("geometric statistics dimension drifted")
        return self.output_head(torch.cat((pooled, statistics), dim=-1)), center


class Stage1CompleteCloudRAE(nn.Module):
    """Point cloud only, with a single shared 128-D decision path."""

    def __init__(
        self,
        *,
        latent_dim: int = 128,
        hidden_dim: int = 768,
        base_feature_dim: int = 512,
        directional_feature_dim: int = 512,
        extra_feature_dim: int = 3,
        translation_residual: float = 0.30,
        reconstruction_points: int = 512,
        query_dim: int = 64,
        coordinate_scale: float = 0.60,
        posterior_std_min: float = 0.02,
        posterior_std_max: float = 0.20,
        posterior_std_init: float = 0.04,
        size_min: float = 0.01,
        size_max: float = 0.80,
        size_init: float = 0.20,
        transformer_layers: int = 4,
        transformer_heads: int = 8,
        base_encoder_version: str = (
            CameraFrameMultiScaleGeometricTransformerEncoder.encoder_version
        ),
        geometric_stage_dim_1: int = 192,
        geometric_stage_dim_2: int = 256,
        geometric_stage_dim_3: int = 384,
        geometric_sample_count_1: int = 384,
        geometric_sample_count_2: int = 192,
        geometric_sample_count_3: int = 96,
        geometric_radius_1: float = 0.12,
        geometric_radius_2: float = 0.25,
        geometric_radius_3: float = 0.45,
        geometric_neighbors: int = 32,
        geometric_local_heads: int = 8,
        directional_token_dim: int = 256,
        directional_token_count: int = 256,
        directional_transformer_layers: int = 4,
        directional_transformer_heads: int = 8,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if latent_dim != 128:
            raise ValueError("Stage1 currently requires a 128-D latent")
        if extra_feature_dim != 3:
            raise ValueError("Stage1 requires one normal per point")
        if min(
            hidden_dim,
            base_feature_dim,
            directional_feature_dim,
            translation_residual,
            reconstruction_points,
            query_dim,
            coordinate_scale,
        ) <= 0:
            raise ValueError("Stage1 dimensions must be positive")
        if not (
            0.0 < posterior_std_min <= posterior_std_init <= posterior_std_max
        ):
            raise ValueError("posterior standard-deviation contract is invalid")
        if not 0.0 < size_min < size_init < size_max:
            raise ValueError("absolute size bounds must contain size_init")

        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.base_feature_dim = int(base_feature_dim)
        self.directional_feature_dim = int(directional_feature_dim)
        self.extra_feature_dim = int(extra_feature_dim)
        self.translation_residual = float(translation_residual)
        self.reconstruction_points = int(reconstruction_points)
        self.query_dim = int(query_dim)
        self.coordinate_scale = float(coordinate_scale)
        self.posterior_std_min = float(posterior_std_min)
        self.posterior_std_max = float(posterior_std_max)
        self.posterior_std_init = float(posterior_std_init)
        self.size_min = float(size_min)
        self.size_max = float(size_max)
        self.size_init = float(size_init)
        self.dropout = float(dropout)
        self.base_encoder_version = str(base_encoder_version)

        if self.base_encoder_version == (
            CameraFrameMultiScaleGeometricTransformerEncoder.encoder_version
        ):
            self.base_encoder = CameraFrameMultiScaleGeometricTransformerEncoder(
                output_dim=self.base_feature_dim,
                stage_dims=(
                    geometric_stage_dim_1,
                    geometric_stage_dim_2,
                    geometric_stage_dim_3,
                ),
                sample_counts=(
                    geometric_sample_count_1,
                    geometric_sample_count_2,
                    geometric_sample_count_3,
                ),
                radii=(
                    geometric_radius_1,
                    geometric_radius_2,
                    geometric_radius_3,
                ),
                neighbors=geometric_neighbors,
                local_heads=geometric_local_heads,
                transformer_layers=transformer_layers,
                transformer_heads=transformer_heads,
                dropout=dropout,
            )
        else:
            raise ValueError(f"unknown class-agnostic base encoder: {base_encoder_version}")
        self.directional_encoder = HighResolutionDirectionalTransformer(
            output_dim=self.directional_feature_dim,
            token_dim=directional_token_dim,
            token_count=directional_token_count,
            transformer_layers=directional_transformer_layers,
            transformer_heads=directional_transformer_heads,
            dropout=dropout,
        )
        self.condition = nn.Sequential(
            nn.Linear(
                self.base_feature_dim + self.directional_feature_dim + 3,
                self.hidden_dim,
            ),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
        )
        self.posterior_mean = _mlp(
            [self.hidden_dim, self.hidden_dim, self.latent_dim], dropout=dropout
        )
        self.posterior_mean_norm = nn.LayerNorm(self.latent_dim)
        self.posterior_scale = _mlp(
            [self.hidden_dim, self.hidden_dim, self.latent_dim], dropout=dropout
        )

        self.pose_trunk = _mlp(
            [self.latent_dim, self.hidden_dim, self.hidden_dim], dropout=dropout
        )
        self.pose_head = nn.Linear(self.hidden_dim, 12)
        self.condition_decoder = _mlp(
            [self.latent_dim, self.hidden_dim, self.hidden_dim], dropout=dropout
        )
        self.point_queries = nn.Parameter(
            torch.randn(self.reconstruction_points, self.query_dim) * 0.02
        )
        self.point_decoder = nn.Sequential(
            nn.Linear(self.latent_dim + self.query_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 3),
        )

        self.register_buffer("identity_rotation_6d", _identity_rotation_6d())
        self.register_buffer(
            "size_reference", torch.full((3,), self.size_init)
        )
        self._reset_outputs()

    def _reset_outputs(self) -> None:
        nn.init.zeros_(self.pose_head.weight)
        nn.init.zeros_(self.pose_head.bias)
        fraction = (self.size_init - self.size_min) / (
            self.size_max - self.size_min
        )
        size_logit = math.log(fraction / (1.0 - fraction))
        with torch.no_grad():
            self.pose_head.bias[:6].copy_(self.identity_rotation_6d)
            self.pose_head.bias[9:12].fill_(size_logit)

        scale_fraction = (
            (self.posterior_std_init - self.posterior_std_min)
            / (self.posterior_std_max - self.posterior_std_min)
            if self.posterior_std_max > self.posterior_std_min
            else 0.5
        )
        scale_fraction = min(max(scale_fraction, 1e-4), 1.0 - 1e-4)
        final = self.posterior_scale[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("posterior scale head must end in Linear")
        nn.init.zeros_(final.weight)
        nn.init.constant_(
            final.bias, math.log(scale_fraction / (1.0 - scale_fraction))
        )

    def encode_condition(
        self,
        points: torch.Tensor,
        *,
        alignment_center: torch.Tensor,
        normals: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError("points must have shape [batch, points, 3]")
        if alignment_center.shape != (points.shape[0], 3):
            raise ValueError("alignment_center must have shape [batch, 3]")
        if normals is None or normals.shape != points.shape:
            raise ValueError("one camera-frame normal is required per point")
        base, full_cloud_center = self.base_encoder(
            points, None, point_features=normals
        )
        directional = self.directional_encoder(points, normals)
        full_to_anchor = (
            full_cloud_center - alignment_center
        ) / self.translation_residual
        condition = self.condition(
            torch.cat((base, directional, full_to_anchor), dim=-1)
        )
        return condition, full_cloud_center, base, directional

    def posterior_from_condition(
        self, condition: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if condition.shape[-1] != self.hidden_dim:
            raise ValueError("encoder condition has the wrong dimension")
        mean = torch.tanh(self.posterior_mean_norm(self.posterior_mean(condition)))
        fraction = torch.sigmoid(self.posterior_scale(condition))
        std = self.posterior_std_min + fraction * (
            self.posterior_std_max - self.posterior_std_min
        )
        log_variance = 2.0 * torch.log(std.clamp_min(1e-8))
        return mean, std, log_variance

    def encode_distribution(
        self,
        points: torch.Tensor,
        *,
        alignment_center: torch.Tensor,
        normals: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        condition, center, base, directional = self.encode_condition(
            points,
            alignment_center=alignment_center,
            normals=normals,
        )
        mean, std, log_variance = self.posterior_from_condition(condition)
        return mean, std, log_variance, condition, center, base, directional

    @staticmethod
    def reparameterize(mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        if mean.shape != std.shape:
            raise ValueError("posterior mean and std must align")
        return mean + std * torch.randn_like(mean)

    def decode_feature(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape[-1] != self.latent_dim:
            raise ValueError("latent has the wrong shape")
        return self.pose_trunk(latent)

    def decode_latent(
        self,
        latent: torch.Tensor,
        *,
        alignment_center: torch.Tensor,
        full_cloud_center: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if alignment_center.shape != (latent.shape[0], 3):
            raise ValueError("alignment_center must have shape [batch, 3]")
        if full_cloud_center is None:
            full_cloud_center = alignment_center
        raw = self.pose_head(self.decode_feature(latent))
        translation = alignment_center + torch.tanh(
            raw[:, 6:9]
        ) * self.translation_residual
        size = self.size_min + (self.size_max - self.size_min) * torch.sigmoid(
            raw[:, 9:12]
        )
        log_size_residual = torch.log(
            size / self.size_reference[None].clamp_min(1e-6)
        )
        return {
            "pose": torch.cat((raw[:, :6], translation), dim=-1),
            "size": size,
            "log_size_residual": log_size_residual,
            "latent": latent,
            "alignment_center": alignment_center,
            "full_cloud_center": full_cloud_center,
            "full_to_anchor_offset": full_cloud_center - alignment_center,
        }

    def reconstruct_condition(self, latent: torch.Tensor) -> torch.Tensor:
        return self.condition_decoder(latent)

    def reconstruct_points(self, latent: torch.Tensor) -> torch.Tensor:
        batch = latent.shape[0]
        query = self.point_queries[None].expand(batch, -1, -1)
        expanded = latent[:, None].expand(-1, self.reconstruction_points, -1)
        relative = self.point_decoder(torch.cat((expanded, query), dim=-1))
        return torch.tanh(relative) * self.coordinate_scale

    def forward(
        self,
        points: torch.Tensor,
        *,
        alignment_center: torch.Tensor,
        normals: torch.Tensor | None = None,
        sample_posterior: bool = False,
        reconstruction_count: int | None = None,
    ) -> dict[str, torch.Tensor]:
        (
            mean,
            std,
            log_variance,
            condition,
            full_cloud_center,
            base,
            directional,
        ) = self.encode_distribution(
            points,
            alignment_center=alignment_center,
            normals=normals,
        )
        prediction = self.decode_latent(
            mean,
            alignment_center=alignment_center,
            full_cloud_center=full_cloud_center,
        )
        if reconstruction_count is None:
            reconstruction_count = mean.shape[0]
        if not 0 <= reconstruction_count <= mean.shape[0]:
            raise ValueError("reconstruction_count lies outside the batch")
        reconstruction_latent = mean[:reconstruction_count]
        prediction.update(
            {
                "latent_mean": mean,
                "latent_std": std,
                "latent_log_variance": log_variance,
                "encoder_condition": condition,
                "base_feature": base,
                "directional_feature": directional,
                "reconstructed_condition": self.reconstruct_condition(reconstruction_latent),
                "reconstructed_points": self.reconstruct_points(reconstruction_latent),
            }
        )
        if sample_posterior:
            sample = self.reparameterize(mean, std)
            sampled = self.decode_latent(
                sample,
                alignment_center=alignment_center,
                full_cloud_center=full_cloud_center,
            )
            prediction.update(
                {
                    "sampled_latent": sample,
                    "sampled_pose": sampled["pose"],
                    "sampled_size": sampled["size"],
                    "sampled_log_size_residual": sampled[
                        "log_size_residual"
                    ],
                }
            )
        return prediction
