"""Train Stage1: complete cloud -> one class-agnostic 128-D RAE latent.

The model receives only camera-frame points, estimated normals, and an anchor
computed from the observed point cloud.  Category labels are used exclusively
by symmetry-aware training losses; they never enter the encoder, latent, pose
decoder, or metric-size decoder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from shared import point_ops
from shared.dataset import get_data_loaders, process_batch
from shared.ema import ExponentialMovingAverage
from shared.point_features import estimate_point_normals
from shared.pose_geometry import canonicalize_continuous_y_rotation, pose6d_to_matrix
from shared.training import (
    DATA_AUGMENTATION,
    MASK_AUGMENTATION,
    POINT_AUGMENTATION,
    augment_complete_cloud,
    chamfer_squared,
    configure_distributed,
    corrupt_complete_cloud,
    mild_complete_view,
    pose_box_loss,
    reduce_epoch_totals,
    seed_everything,
    slice_batch_prediction,
    strict_pose_surrogate,
    symmetric_diagonal_kl,
    symmetry_aware_box_iou_proxy,
)
from stage1.model import (
    CameraFrameMultiScaleGeometricTransformerEncoder,
    Stage1CompleteCloudRAE,
)

CHECKPOINT_FORMAT = "STAGE1_COMPLETE_CLOUD_RAE_V1"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--train_source",
        choices=("CAMERA", "Real", "CAMERA+Real"),
        default="CAMERA+Real",
    )
    parser.add_argument("--epochs", type=int, default=36)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--disable_persistent_workers",
        action="store_true",
        help=(
            "restart data-loader workers at every epoch so a single-process "
            "checkpoint can reconstruct worker RNG streams from the archived "
            "main-process RNG state"
        ),
    )
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--camera_real_ratio", type=float, default=3.0)
    parser.add_argument("--normals_k", type=int, default=16)
    parser.add_argument(
        "--point_ops_backend",
        choices=(point_ops.LEGACY_FPS_BACKEND, point_ops.TORCH_FPS_BACKEND),
        default=point_ops.LEGACY_FPS_BACKEND,
        help=(
            "geometry primitive implementation; legacy_cuda is the original "
            "G10A/Stage1 training contract"
        ),
    )
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--base_feature_dim", type=int, default=512)
    parser.add_argument("--directional_feature_dim", type=int, default=512)
    parser.add_argument("--transformer_layers", type=int, default=4)
    parser.add_argument("--transformer_heads", type=int, default=8)
    parser.add_argument(
        "--base_encoder_version",
        choices=(
            CameraFrameMultiScaleGeometricTransformerEncoder.encoder_version,
        ),
        default=CameraFrameMultiScaleGeometricTransformerEncoder.encoder_version,
    )
    parser.add_argument("--geometric_stage_dim_1", type=int, default=192)
    parser.add_argument("--geometric_stage_dim_2", type=int, default=256)
    parser.add_argument("--geometric_stage_dim_3", type=int, default=384)
    parser.add_argument("--geometric_sample_count_1", type=int, default=384)
    parser.add_argument("--geometric_sample_count_2", type=int, default=192)
    parser.add_argument("--geometric_sample_count_3", type=int, default=96)
    parser.add_argument("--geometric_radius_1", type=float, default=0.12)
    parser.add_argument("--geometric_radius_2", type=float, default=0.25)
    parser.add_argument("--geometric_radius_3", type=float, default=0.45)
    parser.add_argument("--geometric_neighbors", type=int, default=32)
    parser.add_argument("--geometric_local_heads", type=int, default=8)
    parser.add_argument("--directional_token_dim", type=int, default=256)
    parser.add_argument("--directional_token_count", type=int, default=256)
    parser.add_argument("--directional_transformer_layers", type=int, default=4)
    parser.add_argument("--directional_transformer_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--translation_residual", type=float, default=0.30)
    parser.add_argument("--reconstruction_points", type=int, default=512)
    parser.add_argument("--query_dim", type=int, default=64)
    parser.add_argument("--coordinate_scale", type=float, default=0.60)
    parser.add_argument("--posterior_std_min", type=float, default=0.02)
    parser.add_argument("--posterior_std_max", type=float, default=0.20)
    parser.add_argument("--posterior_std_init", type=float, default=0.04)
    parser.add_argument("--size_min", type=float, default=0.01)
    parser.add_argument("--size_max", type=float, default=0.80)
    parser.add_argument("--size_init", type=float, default=0.20)

    parser.add_argument("--base_encoder_lr", type=float, default=2e-4)
    parser.add_argument("--directional_encoder_lr", type=float, default=2.5e-4)
    parser.add_argument("--latent_lr", type=float, default=3e-4)
    parser.add_argument("--decoder_lr", type=float, default=4e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--grad_clip", type=float, default=5.0)

    parser.add_argument("--task_weight", type=float, default=1.0)
    parser.add_argument("--sampled_task_weight", type=float, default=0.25)
    parser.add_argument("--view_task_weight", type=float, default=0.50)
    parser.add_argument("--iou_weight", type=float, default=1.0)
    parser.add_argument("--strict_pose_weight", type=float, default=2.0)
    parser.add_argument(
        "--strict_rotation_temperature_degrees", type=float, default=2.5
    )
    parser.add_argument(
        "--strict_translation_temperature_cm", type=float, default=0.60
    )
    parser.add_argument("--condition_reconstruction_weight", type=float, default=0.15)
    parser.add_argument("--point_reconstruction_weight", type=float, default=20.0)
    parser.add_argument("--view_mean_weight", type=float, default=0.10)
    parser.add_argument("--view_kl_weight", type=float, default=0.01)
    parser.add_argument("--view_base_feature_weight", type=float, default=0.02)
    parser.add_argument(
        "--view_directional_feature_weight", type=float, default=0.05
    )
    parser.add_argument("--posterior_scale_weight", type=float, default=0.01)
    parser.add_argument("--latent_variance_weight", type=float, default=1.0)
    parser.add_argument("--latent_covariance_weight", type=float, default=0.01)
    parser.add_argument("--latent_correlation_weight", type=float, default=0.002)
    parser.add_argument("--latent_mean_weight", type=float, default=0.01)
    parser.add_argument("--latent_std_floor", type=float, default=0.25)
    parser.add_argument("--feature_variance_weight", type=float, default=0.20)
    parser.add_argument("--feature_std_floor", type=float, default=0.20)
    parser.add_argument("--view_contrastive_weight", type=float, default=0.25)
    parser.add_argument("--view_contrastive_temperature", type=float, default=0.10)
    parser.add_argument("--pose_shape_relational_weight", type=float, default=0.50)
    parser.add_argument("--latent_pair_spread_weight", type=float, default=1.0)
    parser.add_argument("--latent_cycle_weight", type=float, default=0.10)
    parser.add_argument("--latent_cycle_fraction", type=float, default=0.125)
    parser.add_argument("--latent_cycle_start_epoch", type=int, default=3)
    parser.add_argument("--size_loss_weight", type=float, default=0.75)

    parser.add_argument("--view_fraction", type=float, default=0.50)
    parser.add_argument("--view_dropout_max", type=float, default=0.18)
    parser.add_argument("--view_noise_std", type=float, default=0.0025)
    parser.add_argument("--noise_max_std", type=float, default=0.006)
    parser.add_argument("--clean_probability", type=float, default=0.25)
    parser.add_argument("--rotation_augmentation_probability", type=float, default=0.50)
    parser.add_argument("--yaw_max_degrees", type=float, default=60.0)
    parser.add_argument("--tilt_probability", type=float, default=0.50)
    parser.add_argument("--tilt_max_degrees", type=float, default=12.0)
    parser.add_argument("--chamfer_chunk", type=int, default=128)
    parser.add_argument("--max_train_batches", type=int, default=-1)
    parser.add_argument("--resume", default="")
    parser.add_argument(
        "--lineage_manifest",
        default="",
        help="optional immutable queue manifest to bind into every checkpoint",
    )
    parser.add_argument(
        "--stop_after_epoch_file",
        default="",
        help=(
            "When this file appears, save the current epoch checkpoint and "
            "exit cleanly so another launcher can resume the run."
        ),
    )
    parser.add_argument("--seed", type=int, default=260892)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--distributed", action="store_true")
    return parser.parse_args()


def architecture(model: Stage1CompleteCloudRAE) -> dict:
    base = model.base_encoder
    directional = model.directional_encoder
    result = {
        "latent_dim": model.latent_dim,
        "hidden_dim": model.hidden_dim,
        "base_feature_dim": model.base_feature_dim,
        "directional_feature_dim": model.directional_feature_dim,
        "extra_feature_dim": model.extra_feature_dim,
        "translation_residual": model.translation_residual,
        "reconstruction_points": model.reconstruction_points,
        "query_dim": model.query_dim,
        "coordinate_scale": model.coordinate_scale,
        "posterior_std_min": model.posterior_std_min,
        "posterior_std_max": model.posterior_std_max,
        "posterior_std_init": model.posterior_std_init,
        "size_min": model.size_min,
        "size_max": model.size_max,
        "size_init": model.size_init,
        "base_encoder_version": model.base_encoder_version,
        "transformer_layers": len(base.global_transformer.layers),
        "transformer_heads": base.global_transformer.layers[0].self_attn.num_heads,
        "directional_token_dim": directional.token_dim,
        "directional_token_count": directional.token_count,
        "directional_transformer_layers": len(directional.transformer.layers),
        "directional_transformer_heads": (
            directional.transformer.layers[0].self_attn.num_heads
        ),
        "dropout": model.dropout,
    }
    if isinstance(base, CameraFrameMultiScaleGeometricTransformerEncoder):
        result.update(
            {
                "geometric_stage_dim_1": base.stage_dims[0],
                "geometric_stage_dim_2": base.stage_dims[1],
                "geometric_stage_dim_3": base.stage_dims[2],
                "geometric_sample_count_1": base.sample_counts[0],
                "geometric_sample_count_2": base.sample_counts[1],
                "geometric_sample_count_3": base.sample_counts[2],
                "geometric_radius_1": base.radii[0],
                "geometric_radius_2": base.radii[1],
                "geometric_radius_3": base.radii[2],
                "geometric_neighbors": base.neighbors,
                "geometric_local_heads": base.local_heads,
            }
        )
    return result


def class_prior_state_audit(
    model: Stage1CompleteCloudRAE,
) -> dict:
    names = tuple(model.state_dict())
    forbidden_fragments = (
        "class_embedding",
        "pose_heads",
        "size_priors",
        "category",
    )
    violations = sorted(
        name
        for name in names
        if any(fragment in name for fragment in forbidden_fragments)
    )
    if violations:
        raise RuntimeError(f"class-prior parameters found: {violations}")
    if model.base_encoder.class_dim != 0 or model.base_encoder.class_embedding is not None:
        raise RuntimeError("base point encoder still contains a class condition")
    return {
        "passed": True,
        "class_embedding": False,
        "category_specific_pose_heads": False,
        "category_size_priors": False,
        "shared_pose_box_decoder": True,
        "class_ids_consumed_by_model": False,
        "state_tensor_count": len(names),
    }


def checkpoint_payload(
    model: Stage1CompleteCloudRAE,
    optimizer,
    scheduler,
    ema,
    epoch: int,
    global_step: int,
    cfg: argparse.Namespace,
    world_size: int,
) -> dict:
    audit = class_prior_state_audit(model)
    lineage_path = (
        os.path.abspath(cfg.lineage_manifest) if cfg.lineage_manifest else ""
    )
    lineage_sha256 = file_sha256(lineage_path) if lineage_path else None
    persistent_workers = bool(
        cfg.num_workers > 0 and not cfg.disable_persistent_workers
    )
    rng_streams_reconstructable = bool(
        int(world_size) == 1 and not persistent_workers
    )
    cuda_runtime = {
        "available": bool(torch.cuda.is_available()),
        "compiled_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "driver_version": os.environ.get("NVIDIA_DRIVER_VERSION"),
    }
    if cfg.device.startswith("cuda") and torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        cuda_runtime.update(
            {
                "visible_device_index": int(device_index),
                "device_name": torch.cuda.get_device_name(device_index),
                "device_capability": list(
                    torch.cuda.get_device_capability(device_index)
                ),
            }
        )
    ema.store(model.parameters())
    ema.copy_to(model.parameters())
    ema_state = {
        key: value.detach().cpu() for key, value in model.state_dict().items()
    }
    ema.restore(model.parameters())
    return {
        "format": CHECKPOINT_FORMAT,
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "ema_model_state_dict": ema_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "ema_state_dict": ema.state_dict(),
        "architecture": architecture(model),
        "class_prior_audit": audit,
        "latent_contract": {
            "dimension": 128,
            "semantics": "fully entangled class-agnostic geometry/pose RAE",
            "decision_path": "point cloud -> multiscale geometric transformer + directional transformer -> 128D -> one shared pose/box decoder",
            "posterior": "bounded diagonal Gaussian",
            "object_type_input": False,
            "absolute_metric_size_from_latent": True,
        },
        "provenance": {
            "trained_from_scratch": True,
            "external_pretrained_weights": False,
            "category_network_input": False,
            "category_used_only_for_symmetry_loss_and_evaluation": True,
            "input": "complete_camera_frame_point_cloud_and_normals_only",
            "encoder": cfg.base_encoder_version,
            "anti_collapse": "VICReg + cross-view InfoNCE + quotient pose/shape relational geometry",
            "train_source": cfg.train_source,
            "full_training": cfg.max_train_batches < 0,
            "point_ops_backend": cfg.point_ops_backend,
        },
        "training_config": vars(cfg),
        "rng_state": capture_rng_state() if int(world_size) == 1 else None,
        "lineage": {
            "manifest_path": lineage_path or None,
            "manifest_sha256": lineage_sha256,
            "resumed_from_checkpoint": bool(cfg.resume),
            "resume_source_path": (
                os.path.abspath(cfg.resume) if cfg.resume else None
            ),
            "resume_source_sha256": cfg.resume_source_sha256,
            "resume_source_epoch": cfg.resume_source_epoch,
            "resume_source_global_step": cfg.resume_source_global_step,
        },
        "execution_protocol": {
            "seed": int(cfg.seed),
            "rank_seed_rule": "seed + distributed rank",
            "world_size": int(world_size),
            "per_rank_batch_size": int(cfg.batch_size),
            "global_batch_size": int(cfg.batch_size) * int(world_size),
            "distributed_data_parallel": bool(world_size > 1),
            "point_ops_backend": cfg.point_ops_backend,
            "rng_state_archived": bool(int(world_size) == 1),
            "rng_state_scope": (
                "python_numpy_torch_cpu_all_visible_cuda"
                if int(world_size) == 1
                else "not_archived_for_distributed_training"
            ),
            "data_loader_workers": int(cfg.num_workers),
            "persistent_data_loader_workers": persistent_workers,
            "worker_rng_reseeded_each_epoch": bool(not persistent_workers),
            "resume_rng_streams_reconstructable": rng_streams_reconstructable,
            "bitwise_determinism_guaranteed": False,
            "python_version": sys.version.split()[0],
            "numpy_version": str(np.__version__),
            "torch_version": str(torch.__version__),
            "cuda_runtime": cuda_runtime,
            "lineage_manifest_sha256": lineage_sha256,
            "physical_gpu_id": os.environ.get("PHYSICAL_GPU"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "resumed_from_checkpoint": bool(cfg.resume),
            "resume_source_sha256": cfg.resume_source_sha256,
            "resume_source_epoch": cfg.resume_source_epoch,
            "resume_source_global_step": cfg.resume_source_global_step,
        },
    }


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_torch_save(payload: dict, path: str) -> None:
    """Publish a checkpoint only after the temporary file is fully written."""

    temporary = f"{path}.tmp.{os.getpid()}"
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    except Exception:
        if os.path.isfile(temporary):
            os.unlink(temporary)
        raise


def capture_rng_state() -> dict:
    """Capture the single-process RNG streams needed for durable continuation."""

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(payload: dict) -> None:
    """Restore a :func:`capture_rng_state` payload before the next epoch."""

    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"checkpoint RNG state is missing fields: {sorted(missing)}")
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["torch_cpu"].cpu())
    if torch.cuda.is_available():
        cuda_states = list(payload["torch_cuda"])
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError("checkpoint CUDA RNG state does not match visible devices")
        torch.cuda.set_rng_state_all(cuda_states)


def normalized_feature_consistency(
    first: torch.Tensor, second: torch.Tensor
) -> torch.Tensor:
    first = F.layer_norm(first, (first.shape[-1],))
    second = F.layer_norm(second, (second.shape[-1],))
    return F.smooth_l1_loss(first, second.detach(), beta=0.25)


def variance_covariance_regularizers(
    value: torch.Tensor, *, std_floor: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """VICReg-style anti-collapse terms computed before any dropout shortcut."""

    if value.ndim != 2 or value.shape[0] < 2:
        raise ValueError("anti-collapse features must have shape [batch>=2, dim]")
    if std_floor <= 0.0:
        raise ValueError("anti-collapse standard-deviation floor must be positive")
    coordinate_mean = value.mean(dim=0)
    centered = value - coordinate_mean
    coordinate_std = torch.sqrt(centered.var(dim=0, unbiased=False) + 1e-4)
    variance = F.relu(std_floor - coordinate_std).mean()
    covariance_matrix = centered.T @ centered / max(1, value.shape[0] - 1)
    covariance = covariance_matrix.square().sum() - covariance_matrix.diag().square().sum()
    covariance = covariance / max(1, value.shape[1] * (value.shape[1] - 1))
    standardized = centered / coordinate_std.clamp_min(1e-4)
    correlation_matrix = standardized.T @ standardized / max(1, value.shape[0] - 1)
    correlation = correlation_matrix.square().sum() - correlation_matrix.diag().square().sum()
    correlation = correlation / max(1, value.shape[1] * (value.shape[1] - 1))
    mean_penalty = coordinate_mean.square().mean()
    return variance, covariance, correlation, mean_penalty


def coordinate_variance_loss(
    value: torch.Tensor, *, std_floor: float
) -> tuple[torch.Tensor, torch.Tensor]:
    if value.ndim != 2 or value.shape[0] < 2:
        raise ValueError("feature variance requires shape [batch>=2, dim]")
    coordinate_std = torch.sqrt(value.var(dim=0, unbiased=False) + 1e-4)
    return F.relu(std_floor - coordinate_std).mean(), coordinate_std.mean().detach()


def view_instance_contrastive_loss(
    primary: torch.Tensor,
    view: torch.Tensor,
    view_indices: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """Instance discrimination prevents view consistency from collapsing."""

    if temperature <= 0.0:
        raise ValueError("contrastive temperature must be positive")
    if view.shape != primary[view_indices].shape:
        raise ValueError("view latents must align with selected primary latents")
    primary_norm = F.normalize(primary, dim=-1, eps=1e-6)
    view_norm = F.normalize(view, dim=-1, eps=1e-6)
    forward_logits = view_norm @ primary_norm.T / temperature
    forward = F.cross_entropy(forward_logits, view_indices)
    reverse_logits = primary_norm[view_indices] @ view_norm.T / temperature
    reverse = F.cross_entropy(
        reverse_logits,
        torch.arange(view.shape[0], device=view.device),
    )
    return 0.5 * (forward + reverse)


def pose_shape_relational_loss(
    latent: torch.Tensor,
    target_pose: torch.Tensor,
    target_size: torch.Tensor,
    class_ids: torch.Tensor,
    handles: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match pairwise latent geometry to quotient pose/box geometry.

    The target relation is computed only from GT pose and metric size.  Class
    labels enter solely to choose the official continuous-symmetry quotient;
    no category embedding or category-dependent target scale is introduced.
    """

    if latent.shape[0] < 2:
        zero = latent.new_zeros(())
        return zero, zero
    rotation = canonicalize_continuous_y_rotation(
        pose6d_to_matrix(target_pose[:, :6]), class_ids, handles
    )
    relation = torch.cat(
        (
            rotation.flatten(1),
            target_pose[:, 6:9] / 0.30,
            torch.log(target_size.clamp_min(1e-5)) / 2.0,
        ),
        dim=-1,
    )
    target_distance = torch.pdist(relation).detach()
    target_distance = target_distance / target_distance.mean().clamp_min(1e-4)
    latent_distance = torch.pdist(latent)
    latent_distance_normalized = latent_distance / latent_distance.mean().detach().clamp_min(1e-4)
    relational = F.smooth_l1_loss(
        latent_distance_normalized,
        target_distance.clamp_max(4.0),
        beta=0.25,
    )
    # Absolute spread is needed because a scale-normalized relational loss
    # alone still permits every latent to approach the same point.
    target_spread = math.sqrt(float(latent.shape[1])) * 0.25
    pair_spread = F.relu(target_spread - latent_distance.mean()) / target_spread
    return relational, pair_spread


def main() -> None:
    cfg = arguments()
    os.environ["INFLUPOSE_FPS_BACKEND"] = cfg.point_ops_backend
    if cfg.epochs <= 0 or cfg.max_train_batches == 0:
        raise ValueError("training length is invalid")
    if cfg.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if cfg.camera_real_ratio <= 0.0:
        raise ValueError("camera_real_ratio must be positive")
    if not 0.0 < cfg.view_fraction <= 1.0:
        raise ValueError("view_fraction must lie in (0, 1]")
    if not 0.0 <= cfg.latent_cycle_fraction <= 1.0:
        raise ValueError("latent_cycle_fraction must lie in [0, 1]")
    if min(
        cfg.latent_std_floor,
        cfg.feature_std_floor,
        cfg.view_contrastive_temperature,
    ) <= 0.0:
        raise ValueError("anti-collapse floors and temperature must be positive")

    distributed, rank, local_rank, world_size = configure_distributed(cfg)
    is_main = rank == 0
    seed_everything(cfg.seed + rank)
    if cfg.device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
    os.makedirs(cfg.output_dir, exist_ok=True)
    cfg.resume_source_sha256 = None
    cfg.resume_source_epoch = None
    cfg.resume_source_global_step = None
    if cfg.lineage_manifest:
        cfg.lineage_manifest = os.path.abspath(cfg.lineage_manifest)
        if not os.path.isfile(cfg.lineage_manifest):
            raise FileNotFoundError(
                f"lineage manifest does not exist: {cfg.lineage_manifest}"
            )

    loader = get_data_loaders(
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        dynamic_zoom_in_params=DATA_AUGMENTATION,
        deform_2d_params=MASK_AUGMENTATION,
        percentage_data=1.0,
        data_path=cfg.data_path,
        source=cfg.train_source,
        mode="train",
        n_pts=cfg.num_points,
        img_size=256,
        num_workers=cfg.num_workers,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        pointcloud_only=True,
        camera_real_ratio=cfg.camera_real_ratio,
        persistent_workers=not cfg.disable_persistent_workers,
    )
    bare_model = Stage1CompleteCloudRAE(
        latent_dim=128,
        hidden_dim=cfg.hidden_dim,
        base_feature_dim=cfg.base_feature_dim,
        directional_feature_dim=cfg.directional_feature_dim,
        extra_feature_dim=3,
        translation_residual=cfg.translation_residual,
        reconstruction_points=cfg.reconstruction_points,
        query_dim=cfg.query_dim,
        coordinate_scale=cfg.coordinate_scale,
        posterior_std_min=cfg.posterior_std_min,
        posterior_std_max=cfg.posterior_std_max,
        posterior_std_init=cfg.posterior_std_init,
        size_min=cfg.size_min,
        size_max=cfg.size_max,
        size_init=cfg.size_init,
        base_encoder_version=cfg.base_encoder_version,
        transformer_layers=cfg.transformer_layers,
        transformer_heads=cfg.transformer_heads,
        geometric_stage_dim_1=cfg.geometric_stage_dim_1,
        geometric_stage_dim_2=cfg.geometric_stage_dim_2,
        geometric_stage_dim_3=cfg.geometric_stage_dim_3,
        geometric_sample_count_1=cfg.geometric_sample_count_1,
        geometric_sample_count_2=cfg.geometric_sample_count_2,
        geometric_sample_count_3=cfg.geometric_sample_count_3,
        geometric_radius_1=cfg.geometric_radius_1,
        geometric_radius_2=cfg.geometric_radius_2,
        geometric_radius_3=cfg.geometric_radius_3,
        geometric_neighbors=cfg.geometric_neighbors,
        geometric_local_heads=cfg.geometric_local_heads,
        directional_token_dim=cfg.directional_token_dim,
        directional_token_count=cfg.directional_token_count,
        directional_transformer_layers=cfg.directional_transformer_layers,
        directional_transformer_heads=cfg.directional_transformer_heads,
        dropout=cfg.dropout,
    ).to(cfg.device)
    audit = class_prior_state_audit(bare_model)

    base_ids = {id(parameter) for parameter in bare_model.base_encoder.parameters()}
    directional_ids = {
        id(parameter) for parameter in bare_model.directional_encoder.parameters()
    }
    latent_modules = (
        bare_model.condition,
        bare_model.posterior_mean,
        bare_model.posterior_mean_norm,
        bare_model.posterior_scale,
    )
    latent_ids = {
        id(parameter)
        for module in latent_modules
        for parameter in module.parameters()
    }
    decoder_parameters = [
        parameter
        for parameter in bare_model.parameters()
        if id(parameter) not in base_ids | directional_ids | latent_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {
                "name": "camera_frame_multiscale_geometric_transformer",
                "params": bare_model.base_encoder.parameters(),
                "lr": cfg.base_encoder_lr,
            },
            {
                "name": "class_agnostic_high_resolution_transformer",
                "params": bare_model.directional_encoder.parameters(),
                "lr": cfg.directional_encoder_lr,
            },
            {
                "name": "class_agnostic_128D_posterior",
                "params": [
                    parameter
                    for module in latent_modules
                    for parameter in module.parameters()
                ],
                "lr": cfg.latent_lr,
            },
            {
                "name": "single_shared_pose_box_geometry_decoder",
                "params": decoder_parameters,
                "lr": cfg.decoder_lr,
            },
        ],
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, cfg.epochs),
        eta_min=min(
            cfg.base_encoder_lr,
            cfg.directional_encoder_lr,
            cfg.latent_lr,
            cfg.decoder_lr,
        )
        * 0.05,
    )
    ema = ExponentialMovingAverage(bare_model.parameters(), decay=cfg.ema_decay)
    start_epoch = 0
    global_step = 0
    if cfg.resume:
        cfg.resume = os.path.abspath(cfg.resume)
        cfg.resume_source_sha256 = file_sha256(cfg.resume)
        state = torch.load(cfg.resume, map_location=cfg.device)
        if state.get("format") != CHECKPOINT_FORMAT:
            raise ValueError("resume is not a Stage1 checkpoint")
        if state.get("architecture") != architecture(bare_model):
            raise ValueError("resume architecture differs")
        if state.get("class_prior_audit", {}).get("passed") is not True:
            raise ValueError("resume checkpoint lacks the class-prior audit")
        bare_model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        ema.load_state_dict(state["ema_state_dict"])
        start_epoch = int(state["epoch"]) + 1
        global_step = int(state["global_step"])
        cfg.resume_source_epoch = int(state["epoch"])
        cfg.resume_source_global_step = global_step
        if world_size == 1:
            rng_state = state.get("rng_state")
            if rng_state is not None:
                restore_rng_state(rng_state)
                if (
                    cfg.num_workers > 0
                    and not cfg.disable_persistent_workers
                    and is_main
                ):
                    print(
                        "WARNING: main-process RNG state was restored, but "
                        "persistent data-loader worker RNG streams are not "
                        "checkpointed",
                        flush=True,
                    )
            elif is_main:
                print(
                    "WARNING: legacy Stage1 resume lacks RNG state; continuation "
                    "is protocol-compatible but not RNG-exact",
                    flush=True,
                )

    model: torch.nn.Module = (
        DistributedDataParallel(
            bare_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
        if distributed
        else bare_model
    )
    history_path = os.path.join(cfg.output_dir, "history.jsonl")
    if is_main:
        print(
            json.dumps(
                {
                    "run": "stage1-complete-cloud-rae128",
                    "checkpoint_format": CHECKPOINT_FORMAT,
                    "base_encoder_version": cfg.base_encoder_version,
                    "latent_dim": 128,
                    "class_prior_audit": audit,
                    "external_pretrained_weights": False,
                    "full_training": cfg.max_train_batches < 0,
                    "train_source": cfg.train_source,
                    "world_size": world_size,
                    "global_batch_size": cfg.batch_size * world_size,
                    "rgb_enabled": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        if distributed and hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)
        totals: dict[str, float] = {}
        examples = 0
        progress = (
            tqdm(loader, desc=f"Stage1 RAE128 {epoch + 1}/{cfg.epochs}")
            if is_main
            else loader
        )
        for batch_index, raw in enumerate(progress):
            if 0 <= cfg.max_train_batches <= batch_index:
                break
            batch = process_batch(
                raw,
                device=cfg.device,
                pose_mode="rot_matrix",
                PTS_AUG_PARAMS=POINT_AUGMENTATION,
                return_full_cloud=True,
                full_cloud_use_dense=False,
            )
            points = corrupt_complete_cloud(
                batch["full_cloud"],
                max_std=cfg.noise_max_std,
                clean_probability=cfg.clean_probability,
                epoch=epoch,
                warmup_epochs=0,
            )
            class_ids = batch["id"].long()
            handles = batch["handle_visibility"].long()
            target_pose = batch["gt_pose"]
            target_size = batch["gt_size"]
            alignment_center = batch["pts_center"]
            points, target_pose, alignment_center = augment_complete_cloud(
                points,
                target_pose,
                alignment_center,
                rotation_probability=cfg.rotation_augmentation_probability,
                yaw_max_degrees=cfg.yaw_max_degrees,
                tilt_probability=cfg.tilt_probability,
                tilt_max_degrees=cfg.tilt_max_degrees,
            )
            normals = estimate_point_normals(
                points, points.mean(dim=1), k=cfg.normals_k
            ).detach()

            batch_size = points.shape[0]
            view_count = max(1, round(batch_size * cfg.view_fraction))
            view_indices = torch.randperm(batch_size, device=points.device)[:view_count]
            view_points = mild_complete_view(
                points[view_indices],
                dropout_max=cfg.view_dropout_max,
                noise_std=cfg.view_noise_std,
            )
            view_normals = estimate_point_normals(
                view_points, view_points.mean(dim=1), k=cfg.normals_k
            ).detach()
            combined_count = batch_size + view_count
            combined = model(
                torch.cat((points, view_points), dim=0),
                alignment_center=torch.cat(
                    (alignment_center, alignment_center[view_indices]), dim=0
                ),
                normals=torch.cat((normals, view_normals), dim=0),
                sample_posterior=True,
                reconstruction_count=batch_size,
            )
            prediction = slice_batch_prediction(combined, batch_size, combined_count)
            view_prediction = {
                "pose": combined["pose"][batch_size:],
                "size": combined["size"][batch_size:],
            }
            sampled_prediction = {
                "pose": prediction["sampled_pose"],
                "size": prediction["sampled_size"],
            }

            task, task_stats = pose_box_loss(
                prediction,
                target_pose,
                target_size,
                class_ids,
                handles,
                cfg.size_loss_weight,
            )
            sampled_task, _ = pose_box_loss(
                sampled_prediction,
                target_pose,
                target_size,
                class_ids,
                handles,
                cfg.size_loss_weight,
            )
            view_task, _ = pose_box_loss(
                view_prediction,
                target_pose[view_indices],
                target_size[view_indices],
                class_ids[view_indices],
                handles[view_indices],
                cfg.size_loss_weight,
            )
            box_iou = symmetry_aware_box_iou_proxy(
                prediction["pose"],
                prediction["size"],
                target_pose,
                target_size,
                class_ids,
                handles,
            )
            iou_loss = 1.0 - box_iou.mean()
            strict_pose, strict_success = strict_pose_surrogate(
                prediction,
                target_pose,
                class_ids,
                handles,
                rotation_temperature_degrees=(
                    cfg.strict_rotation_temperature_degrees
                ),
                translation_temperature_cm=(
                    cfg.strict_translation_temperature_cm
                ),
            )

            condition_target = F.layer_norm(
                prediction["encoder_condition"].detach(),
                (prediction["encoder_condition"].shape[-1],),
            )
            reconstructed_condition = F.layer_norm(
                prediction["reconstructed_condition"],
                (prediction["reconstructed_condition"].shape[-1],),
            )
            condition_reconstruction = F.smooth_l1_loss(
                reconstructed_condition, condition_target, beta=0.25
            ) + 0.10 * (
                1.0
                - F.cosine_similarity(
                    reconstructed_condition, condition_target, dim=-1
                )
            ).mean()
            target_relative = points - alignment_center[:, None]
            point_reconstruction = chamfer_squared(
                prediction["reconstructed_points"],
                target_relative,
                chunk_size=cfg.chamfer_chunk,
            ).mean()

            primary_mean = prediction["latent_mean"]
            primary_std = prediction["latent_std"]
            view_mean = combined["latent_mean"][batch_size:]
            view_std = combined["latent_std"][batch_size:]
            mean_consistency = F.smooth_l1_loss(
                view_mean, primary_mean[view_indices].detach(), beta=0.10
            )
            distribution_consistency = symmetric_diagonal_kl(
                primary_mean[view_indices],
                primary_std[view_indices],
                view_mean,
                view_std,
            )
            base_consistency = normalized_feature_consistency(
                combined["base_feature"][batch_size:],
                prediction["base_feature"][view_indices],
            )
            directional_consistency = normalized_feature_consistency(
                combined["directional_feature"][batch_size:],
                prediction["directional_feature"][view_indices],
            )
            posterior_scale = F.smooth_l1_loss(
                torch.log(primary_std.clamp_min(1e-8)),
                torch.full_like(primary_std, math.log(cfg.posterior_std_init)),
                beta=0.25,
            )
            variance, covariance, correlation, latent_mean = (
                variance_covariance_regularizers(
                    primary_mean, std_floor=cfg.latent_std_floor
                )
            )
            latent_std = torch.sqrt(
                primary_mean.var(dim=0, unbiased=False) + 1e-4
            ).mean().detach()
            base_variance, base_std = coordinate_variance_loss(
                prediction["base_feature"], std_floor=cfg.feature_std_floor
            )
            directional_variance, directional_std = coordinate_variance_loss(
                prediction["directional_feature"],
                std_floor=cfg.feature_std_floor,
            )
            condition_variance, condition_std = coordinate_variance_loss(
                prediction["encoder_condition"],
                std_floor=cfg.feature_std_floor,
            )
            feature_variance = (
                base_variance + directional_variance + condition_variance
            ) / 3.0
            view_contrastive = view_instance_contrastive_loss(
                primary_mean,
                view_mean,
                view_indices,
                temperature=cfg.view_contrastive_temperature,
            )
            pose_shape_relational, latent_pair_spread = pose_shape_relational_loss(
                primary_mean,
                target_pose,
                target_size,
                class_ids,
                handles,
            )

            latent_cycle = primary_mean.new_zeros(())
            if (
                epoch >= cfg.latent_cycle_start_epoch
                and cfg.latent_cycle_weight > 0.0
                and cfg.latent_cycle_fraction > 0.0
            ):
                cycle_count = max(
                    1, round(batch_size * cfg.latent_cycle_fraction)
                )
                cycle_mean = bare_model.posterior_from_condition(
                    prediction["reconstructed_condition"][:cycle_count]
                )[0]
                latent_cycle = F.smooth_l1_loss(
                    cycle_mean,
                    primary_mean[:cycle_count].detach(),
                    beta=0.10,
                )

            strict_ramp = min(1.0, (epoch + 1) / 5.0)
            cycle_ramp = min(
                1.0,
                max(0.0, (epoch - cfg.latent_cycle_start_epoch + 1) / 4.0),
            )
            loss = (
                cfg.task_weight * task
                + cfg.sampled_task_weight * sampled_task
                + cfg.view_task_weight * view_task
                + cfg.iou_weight * iou_loss
                + strict_ramp * cfg.strict_pose_weight * strict_pose
                + cfg.condition_reconstruction_weight * condition_reconstruction
                + cfg.point_reconstruction_weight * point_reconstruction
                + cfg.view_mean_weight * mean_consistency
                + cfg.view_kl_weight * distribution_consistency
                + cfg.view_base_feature_weight * base_consistency
                + cfg.view_directional_feature_weight * directional_consistency
                + cfg.posterior_scale_weight * posterior_scale
                + cfg.latent_variance_weight * variance
                + cfg.latent_covariance_weight * covariance
                + cfg.latent_correlation_weight * correlation
                + cfg.latent_mean_weight * latent_mean
                + cfg.feature_variance_weight * feature_variance
                + cfg.view_contrastive_weight * view_contrastive
                + cfg.pose_shape_relational_weight * pose_shape_relational
                + cfg.latent_pair_spread_weight * latent_pair_spread
                + cycle_ramp * cfg.latent_cycle_weight * latent_cycle
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bare_model.parameters(), cfg.grad_clip)
            optimizer.step()
            ema.update(bare_model.parameters())
            global_step += 1

            count = int(batch_size)
            examples += count
            values = {
                "loss": loss.detach(),
                "task": task.detach(),
                "sampled_task": sampled_task.detach(),
                "view_task": view_task.detach(),
                "iou_loss": iou_loss.detach(),
                "iou50": (box_iou >= 0.50).float().mean().detach(),
                "iou75": (box_iou >= 0.75).float().mean().detach(),
                "strict_pose_surrogate": strict_pose.detach(),
                "strict_success": strict_success.detach(),
                "condition_reconstruction": condition_reconstruction.detach(),
                "point_chamfer_m2": point_reconstruction.detach(),
                "view_mean": mean_consistency.detach(),
                "view_kl": distribution_consistency.detach(),
                "view_base": base_consistency.detach(),
                "view_directional": directional_consistency.detach(),
                "posterior_scale": posterior_scale.detach(),
                "latent_variance": variance.detach(),
                "latent_covariance": covariance.detach(),
                "latent_correlation": correlation.detach(),
                "latent_mean": latent_mean.detach(),
                "latent_std": latent_std.detach(),
                "base_std": base_std,
                "directional_std": directional_std,
                "condition_std": condition_std,
                "feature_variance": feature_variance.detach(),
                "view_contrastive": view_contrastive.detach(),
                "pose_shape_relational": pose_shape_relational.detach(),
                "latent_pair_spread": latent_pair_spread.detach(),
                "latent_cycle": latent_cycle.detach(),
                **task_stats,
            }
            for name, value in values.items():
                totals[name] = totals.get(name, 0.0) + float(value) * count
            if is_main:
                progress.set_postfix(
                    loss=f"{float(loss):.3f}",
                    hit=f"{float(strict_success):.2f}",
                    r=f"{float(task_stats['rotation_deg']):.1f}",
                    t=f"{float(task_stats['translation_cm']):.2f}",
                )

        scheduler.step()
        totals, examples = reduce_epoch_totals(
            totals, examples, device=cfg.device, distributed=distributed
        )
        summary = {
            "epoch": epoch,
            "stage": "stage1_complete_cloud_rae128",
            "examples": examples,
            "global_step": global_step,
            "learning_rates": [group["lr"] for group in optimizer.param_groups],
            **{name: value / max(1, examples) for name, value in totals.items()},
        }
        if is_main:
            with open(history_path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(summary, sort_keys=True) + "\n")
            print(json.dumps(summary, sort_keys=True), flush=True)
            payload = checkpoint_payload(
                bare_model,
                optimizer,
                scheduler,
                ema,
                epoch,
                global_step,
                cfg,
                world_size,
            )
            atomic_torch_save(
                payload,
                os.path.join(cfg.output_dir, f"epoch_{epoch + 1:03d}.pth"),
            )
            atomic_torch_save(
                payload,
                os.path.join(cfg.output_dir, "latest.pth"),
            )
        if distributed:
            torch.distributed.barrier()

        stop_requested = bool(
            is_main
            and cfg.stop_after_epoch_file
            and os.path.isfile(cfg.stop_after_epoch_file)
        )
        if distributed:
            stop_flag = torch.tensor(
                int(stop_requested), device=cfg.device, dtype=torch.int32
            )
            torch.distributed.broadcast(stop_flag, src=0)
            stop_requested = bool(stop_flag.item())
        if stop_requested:
            if is_main:
                print(
                    json.dumps(
                        {
                            "stage1_handoff": "requested",
                            "completed_epochs": epoch + 1,
                            "checkpoint": os.path.join(
                                cfg.output_dir, "latest.pth"
                            ),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            break

    if distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
