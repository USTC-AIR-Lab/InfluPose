"""Pose-sensitive high-resolution point encoder used by Stage1."""

from __future__ import annotations

import torch
import torch.nn as nn

from shared import point_ops


class HighResolutionDirectionalTransformer(nn.Module):
    """Preserve signed camera-frame evidence at 256 point tokens.

    The module intentionally has no category input.  Its parameter layout is
    identical to the encoder stored in the archived Stage1 checkpoint.
    """

    def __init__(
        self,
        *,
        output_dim: int = 512,
        token_dim: int = 256,
        token_count: int = 256,
        transformer_layers: int = 4,
        transformer_heads: int = 8,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if min(
            output_dim,
            token_dim,
            token_count,
            transformer_layers,
            transformer_heads,
        ) <= 0:
            raise ValueError("directional transformer dimensions must be positive")
        if token_dim % transformer_heads:
            raise ValueError("directional token_dim must divide transformer_heads")
        self.output_dim = int(output_dim)
        self.token_dim = int(token_dim)
        self.token_count = int(token_count)
        self.input_embedding = nn.Sequential(
            nn.Linear(14, self.token_dim),
            nn.LayerNorm(self.token_dim),
            nn.SiLU(),
            nn.Linear(self.token_dim, self.token_dim),
        )
        self.position_embedding = nn.Sequential(
            nn.Linear(3, self.token_dim),
            nn.LayerNorm(self.token_dim),
            nn.SiLU(),
            nn.Linear(self.token_dim, self.token_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.token_dim,
            nhead=transformer_heads,
            dim_feedforward=4 * self.token_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=transformer_layers,
            norm=nn.LayerNorm(self.token_dim),
        )
        # bounds(6), covariance(9), signed third moment(27),
        # position-normal cross moment(9), mean normal(3), radial skew(3).
        self.statistics_dim = 57
        self.output_head = nn.Sequential(
            nn.Linear(2 * self.token_dim + self.statistics_dim, 768),
            nn.LayerNorm(768),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(768, self.output_dim),
            nn.LayerNorm(self.output_dim),
        )

    @staticmethod
    def _gather(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return point_ops.gather_operation(
            values.transpose(1, 2).contiguous(), indices
        ).transpose(1, 2).contiguous()

    def encode_tokens(
        self,
        points: torch.Tensor,
        normals: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Expose the same tokens used by the pooled forward path."""

        feature, details = self._encode(points, normals)
        details["feature"] = feature
        return details

    def _encode(
        self,
        points: torch.Tensor,
        normals: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError("directional points must have shape [batch, points, 3]")
        if normals.shape != points.shape:
            raise ValueError("directional normals must align with points")
        center = points.mean(dim=1)
        relative = points - center[:, None]
        extent = relative.amax(dim=1) - relative.amin(dim=1)
        scale = extent.amax(dim=-1, keepdim=True).clamp_min(0.03)
        normalized = relative / scale[:, None]
        normals = torch.nn.functional.normalize(normals, dim=-1, eps=1e-6)
        cross = torch.cross(normalized, normals, dim=-1)
        dot = (normalized * normals).sum(dim=-1, keepdim=True)
        radius = torch.linalg.vector_norm(normalized, dim=-1, keepdim=True)
        point_feature = torch.cat(
            (normalized, normals, cross, dot, radius, normalized * radius), dim=-1
        )
        sample_count = min(self.token_count, points.shape[1])
        indices = point_ops.farthest_point_sample(
            normalized.contiguous(), sample_count
        )
        token_xyz = self._gather(normalized, indices)
        token_feature = self._gather(point_feature, indices)
        tokens = self.input_embedding(token_feature)
        tokens = tokens + self.position_embedding(token_xyz)
        tokens = self.transformer(tokens)
        pooled = torch.cat((tokens.amax(dim=1), tokens.mean(dim=1)), dim=-1)

        covariance = torch.einsum(
            "bni,bnj->bij", normalized, normalized
        ) / max(1, normalized.shape[1])
        third = torch.einsum(
            "bni,bnj,bnk->bijk", normalized, normalized, normalized
        ) / max(1, normalized.shape[1])
        position_normal = torch.einsum(
            "bni,bnj->bij", normalized, normals
        ) / max(1, normalized.shape[1])
        radial_skew = (normalized * radius.square()).mean(dim=1)
        statistics = torch.cat(
            (
                normalized.amin(dim=1),
                normalized.amax(dim=1),
                covariance.flatten(1),
                third.flatten(1),
                position_normal.flatten(1),
                normals.mean(dim=1),
                radial_skew,
            ),
            dim=-1,
        )
        if statistics.shape[-1] != self.statistics_dim:
            raise RuntimeError("directional statistics contract changed")
        feature = self.output_head(torch.cat((pooled, statistics), dim=-1))
        return feature, {
            "tokens": tokens,
            "token_xyz": token_xyz,
            "statistics": statistics,
            "center": center,
            "scale": scale,
        }

    def forward(self, points: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
        feature, _ = self._encode(points, normals)
        return feature
