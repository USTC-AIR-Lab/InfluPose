"""Train the partial-cloud predictor with complete-geometry influence targets.

The complete-cloud teacher remains frozen throughout training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from shared.execution_protocol import (
    EXECUTION_PROTOCOL_FORMAT,
    checkpoint_execution_layout,
    record_runtime_execution_layout,
)
from shared import point_ops
from shared.dataset import get_data_loaders, process_batch
from shared.ema import ExponentialMovingAverage
from shared.point_features import (
    estimate_camera_observation_features,
    estimate_point_normals,
)
from shared.pose_geometry import (
    continuous_y_symmetry_mask,
    pose6d_to_matrix,
)
from shared.training import (
    DATA_AUGMENTATION,
    MASK_AUGMENTATION,
    POINT_AUGMENTATION,
    configure_distributed,
    pose_box_loss,
    reduce_epoch_totals,
    seed_everything,
    symmetry_aware_box_iou_proxy,
)
from stage1.checkpoint import load_model as load_stage1
from stage2.auxiliary_losses import (
    AUXILIARY_DEFAULTS,
    detached_state_dict,
    forward_training_views,
    frozen_reference,
    reference_non_regression_loss,
    validate_auxiliary_config,
)
from stage2.model import Stage2PoseInformationField
from stage2.schedule import begin_refinement, make_optimizer, make_scheduler as make_stage2_scheduler
from stage2.observability import (
    analytic_correspondence_information,
    regional_stage1_observability,
    transfer_region_information,
)

CHECKPOINT_FORMAT = "STAGE2_G10F_INFORMATION_FIELD_IRLS_V1"
TRAINING_PROTOCOL = "RANDOM_INIT_BASE30_REFINE20_V6"
REFINEMENT_DEFAULTS = {
    "refine_orientation_canonical_weight": 2.5,
    "refine_view_pose_weight": 1.0,
}


def solver_contract(iterations: int) -> str:
    """Describe the deterministic endpoint without changing release behavior."""

    updates = int(iterations)
    if updates < 0:
        raise ValueError("IRLS iterations must be non-negative")
    if updates == 0:
        return "inlier_precision_influence_weighted_kabsch_only"
    label = "three" if updates == 3 else str(updates)
    return f"{label}_iteration_anisotropic_information_weighted_irls"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--stage1_checkpoint", default="checkpoints/stage1.pth")
    parser.add_argument(
        "--observability_audit", default="checkpoints/stage1_observability_audit.json"
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_source", choices=("CAMERA", "Real", "CAMERA+Real"), default="CAMERA+Real")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--base_epochs", type=int, default=30)
    parser.add_argument("--refine_learning_rate", type=float, default=1.5e-5)
    parser.add_argument("--refine_head_learning_rate", type=float, default=3e-5)
    parser.add_argument("--refine_eta_min", type=float, default=3e-7)
    parser.add_argument("--refine_camera_real_ratio", type=float, default=0.75)
    for name, default in REFINEMENT_DEFAULTS.items():
        parser.add_argument(f"--{name}", type=float, default=default)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--normals_k", type=int, default=16)
    parser.add_argument(
        "--fps_backend",
        choices=(point_ops.TORCH_FPS_BACKEND, point_ops.LEGACY_FPS_BACKEND),
        default=point_ops.LEGACY_FPS_BACKEND,
        help=(
            "point sampling implementation; legacy_cuda is the canonical "
            "reference Stage2 numerical contract"
        ),
    )
    parser.add_argument("--camera_real_ratio", type=float, default=1.0)
    parser.add_argument("--max_train_batches", type=int, default=-1)

    parser.add_argument("--token_width", type=int, default=192)
    parser.add_argument("--anchor_count", type=int, default=192)
    parser.add_argument("--relation_layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--canonical_range", type=float, default=0.65)
    parser.add_argument("--irls_iterations", type=int, default=3)
    parser.add_argument("--irls_damping", type=float, default=1e-3)
    parser.add_argument("--max_rotation_step_degrees", type=float, default=8.0)
    parser.add_argument("--max_translation_step", type=float, default=0.015)
    parser.add_argument("--max_log_size_step", type=float, default=0.12)

    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--head_learning_rate", type=float, default=7e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--amp", choices=("none", "bfloat16"), default="bfloat16")

    parser.add_argument("--pose_weight", type=float, default=2.0)
    parser.add_argument("--iou_weight", type=float, default=1.0)
    parser.add_argument("--canonical_weight", type=float, default=8.0)
    parser.add_argument("--relation_weight", type=float, default=1.0)
    parser.add_argument("--inlier_weight", type=float, default=1.0)
    parser.add_argument("--uncertainty_weight", type=float, default=0.5)
    parser.add_argument("--information_regression_weight", type=float, default=1.0)
    parser.add_argument("--information_kl_weight", type=float, default=1.0)
    parser.add_argument("--information_mass_weight", type=float, default=0.25)
    parser.add_argument("--initial_size_weight", type=float, default=1.0)
    parser.add_argument("--size_loss_weight", type=float, default=0.75)
    for name, default in AUXILIARY_DEFAULTS.items():
        parser.add_argument(f"--{name}", type=type(default), default=default)
    parser.add_argument("--outlier_ratio", type=float, default=0.04)
    parser.add_argument("--outlier_extent", type=float, default=1.8)
    parser.add_argument("--observability_interval", type=int, default=2)
    parser.add_argument("--observability_fraction", type=float, default=0.50)
    parser.add_argument("--observability_regions", type=int, default=8)
    parser.add_argument("--observability_removal_fraction", type=float, default=0.125)
    parser.add_argument("--observability_teacher_chunk", type=int, default=24)
    parser.add_argument("--teacher_label_blend", type=float, default=0.70)
    parser.add_argument("--information_temperature", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=260907)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--resume", default="")
    parser.add_argument(
        "--allow_noncanonical_schedule",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_data_path(path: str) -> str:
    """Resolve a dataset root for checkpoint-lineage comparisons."""

    return os.path.realpath(os.path.abspath(os.fspath(path)))


def require_same_training_dataset(
    source_config: dict[str, object],
    cfg: argparse.Namespace,
    *,
    context: str,
) -> None:
    """Reject initialization or resume state produced from another dataset."""

    source_path = source_config.get("data_path")
    if not source_path or canonical_data_path(str(source_path)) != canonical_data_path(
        cfg.data_path
    ):
        raise ValueError(f"{context} belongs to another training dataset")
    if source_config.get("train_source") != cfg.train_source:
        raise ValueError(f"{context} changed the CAMERA/Real training source")


def inject_outliers(
    points: torch.Tensor, ratio: float, extent_factor: float
) -> tuple[torch.Tensor, torch.Tensor]:
    if ratio <= 0.0:
        return points, torch.ones(points.shape[:2], dtype=torch.bool, device=points.device)
    center = points.mean(dim=1, keepdim=True)
    extent = (points.amax(dim=1) - points.amin(dim=1)).clamp_min(0.03)
    mask = torch.rand(points.shape[:2], device=points.device) < ratio
    random_xyz = center + (torch.rand_like(points) - 0.5) * extent[:, None] * extent_factor
    return torch.where(mask[..., None], random_xyz, points), ~mask


def inverse_pose_coordinates(
    points: torch.Tensor, pose: torch.Tensor, size: torch.Tensor
) -> torch.Tensor:
    rotation = pose6d_to_matrix(pose[:, :6])
    metric = torch.einsum(
        "bij,bnj->bni", rotation.transpose(-1, -2), points - pose[:, None, 6:]
    )
    return metric / size[:, None].clamp_min(1e-4)


def gather_rows(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return torch.gather(
        values, 1, indices[..., None].expand(-1, -1, values.shape[-1])
    )


def coordinate_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    symmetric: torch.Tensor,
) -> torch.Tensor:
    ordinary = F.smooth_l1_loss(prediction, target, beta=0.03, reduction="none").mean(dim=-1)
    pred_radius = torch.linalg.vector_norm(prediction[..., (0, 2)], dim=-1)
    target_radius = torch.linalg.vector_norm(target[..., (0, 2)], dim=-1)
    quotient = F.smooth_l1_loss(
        prediction[..., 1], target[..., 1], beta=0.03, reduction="none"
    ) + F.smooth_l1_loss(pred_radius, target_radius, beta=0.03, reduction="none")
    return torch.where(symmetric[:, None], quotient, ordinary)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    numeric = mask.to(values.dtype)
    return (values * numeric).sum() / numeric.sum().clamp_min(1.0)


def relation_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    predicted_size: torch.Tensor,
    target_size: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    indices = torch.arange(0, prediction.shape[1], 4, device=prediction.device)
    pred_distance = torch.cdist(
        prediction[:, indices] * predicted_size[:, None],
        prediction[:, indices] * predicted_size[:, None],
    )
    target_distance = torch.cdist(
        target[:, indices] * target_size[:, None],
        target[:, indices] * target_size[:, None],
    )
    pair = valid[:, indices, None] & valid[:, None, indices]
    return masked_mean(
        F.smooth_l1_loss(pred_distance, target_distance, beta=0.005, reduction="none"),
        pair,
    )


def information_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    symmetric: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    channel_mask = torch.ones(
        prediction.shape[0], 1, 4, device=prediction.device, dtype=prediction.dtype
    )
    channel_mask[symmetric, :, 1] = 0.0
    point_mask = valid[..., None].to(prediction.dtype)
    active = point_mask * channel_mask
    regression = masked_mean(
        F.smooth_l1_loss(
            torch.log(prediction.clamp_min(1e-5)),
            torch.log(target.clamp_min(1e-5)),
            beta=0.20,
            reduction="none",
        ),
        active,
    )
    invalid = ~valid
    target_logits = torch.log(target.clamp_min(1e-5)) / temperature
    prediction_logits = torch.log(prediction.clamp_min(1e-5)) / temperature
    target_logits = target_logits.masked_fill(invalid[..., None], -1e4)
    prediction_logits = prediction_logits.masked_fill(invalid[..., None], -1e4)
    target_probability = F.softmax(target_logits, dim=1)
    target_log_probability = F.log_softmax(target_logits, dim=1)
    prediction_log_probability = F.log_softmax(prediction_logits, dim=1)
    kl_per_channel = (
        target_probability * (target_log_probability - prediction_log_probability)
    ).sum(dim=1)
    kl = masked_mean(kl_per_channel, channel_mask[:, 0].bool())
    pred_mass = masked_mean(prediction, active)
    target_mass = masked_mean(target, active)
    mass = F.smooth_l1_loss(pred_mass, target_mass, beta=0.10)
    return regression, kl, mass


def class_prior_audit(model: Stage2PoseInformationField) -> dict[str, object]:
    forbidden = ("class_embedding", "category_embedding", "class_head", "category_head")
    violations = sorted(
        name for name in model.state_dict() if any(token in name for token in forbidden)
    )
    if violations:
        raise RuntimeError(f"Stage2 contains forbidden class-conditioned state: {violations}")
    return {
        "passed": True,
        "class_id_network_input": False,
        "class_embedding": False,
        "category_specific_head": False,
        "category_used_only_for_symmetry_loss_and_solver_gauge": True,
        "single_deterministic_endpoint": True,
        "state_tensor_count": len(model.state_dict()),
    }


@torch.no_grad()
def contract_audit(model: Stage2PoseInformationField, device: str) -> dict[str, object]:
    was_training = model.training
    model.eval()
    points = torch.randn(2, 256, 3, device=device) * 0.05
    points[..., 2] += 0.8
    center = points.mean(dim=1)
    normals, observation = estimate_camera_observation_features(points, center, k=8)
    output = model(
        points,
        normals,
        observation,
        alignment_center=center,
        continuous_symmetry=torch.tensor((False, True), device=device),
    )
    finite = all(
        torch.isfinite(value).all()
        for value in output.values()
        if torch.is_tensor(value)
    )
    passed = bool(
        finite
        and output["pose"].shape == (2, 9)
        and output["size"].shape == (2, 3)
        and output["canonical"].shape[0] == 2
        and output["information"].shape[-1] == 4
    )
    if was_training:
        model.train()
    if not passed:
        raise RuntimeError("Stage2 deterministic endpoint contract audit failed")
    return {
        "passed": True,
        "finite": True,
        "pose_shape": list(output["pose"].shape),
        "information_shape": list(output["information"].shape),
        "initial_pose_shape": [
            output["initial_rotation"].shape[0],
            9,
        ],
        "irls_iterations": model.irls_iterations,
        "final_pose_solver": solver_contract(model.irls_iterations),
    }


def checkpoint_payload(
    model: Stage2PoseInformationField,
    optimizer,
    scheduler,
    ema,
    epoch: int,
    global_step: int,
    cfg: argparse.Namespace,
    stage1_sha: str,
    audit: dict,
    reference_state: dict[str, torch.Tensor] | None,
) -> dict:
    ema.store(model.parameters())
    ema.copy_to(model.parameters())
    ema_state = detached_state_dict(model)
    ema.restore(model.parameters())
    return {
        "format": CHECKPOINT_FORMAT,
        "training_protocol": TRAINING_PROTOCOL,
        "reference_model_state_dict": reference_state,
        "reference_policy": "frozen_base_endpoint_ema",
        "reference_epoch": cfg.base_epochs if reference_state is not None else None,
        "training_phase": "base" if epoch < cfg.base_epochs else "refinement",
        "epoch": epoch,
        "epoch_index_base": 0,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "ema_model_state_dict": ema_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "ema_state_dict": ema.state_dict(),
        "recommended_evaluation_weights": "ema",
        "architecture": model.architecture(),
        "class_prior_audit": class_prior_audit(model),
        "contract_audit": audit,
        "stage1_checkpoint": os.path.abspath(cfg.stage1_checkpoint),
        "stage1_checkpoint_sha256": stage1_sha,
        "observability_audit": os.path.abspath(cfg.observability_audit),
        "training_config": dict(vars(cfg)),
        "provenance": {
            "uses_rgb": False,
            "pointcloud_only": True,
            "external_pretrained_weights": False,
            "generative_completion": False,
            "latent_prediction": False,
            "stage1_training_teacher": True,
            "stage1_at_inference": False,
            "category_network_input": False,
            "single_deterministic_pose": True,
            "final_solver": solver_contract(cfg.irls_iterations),
            "full_dataset_indexed": True,
            "full_training": cfg.max_train_batches < 0,
            "random_lineage": True,
            "run_started_from_random_weights": True,
            "fps_backend": cfg.fps_backend,
            "base_epochs": cfg.base_epochs,
            "refinement_epochs": cfg.epochs - cfg.base_epochs,
            "external_stage2_initialization": False,
            "same_stage_resume": bool(cfg.resume),
            "execution_protocol_format": EXECUTION_PROTOCOL_FORMAT,
            "execution_layout": cfg.execution_layout,
            "resolved_world_size": cfg.resolved_world_size,
            "resolved_global_batch_size": cfg.resolved_global_batch_size,
            "per_device_batch_size": cfg.batch_size,
            "bit_exact_equivalence_to_other_execution_layout_claimed": False,
            "execution_numerics_note": (
                "the single-GPU and legacy six-way DDP layouts preserve the "
                "global-batch/epoch/step/learning-rate contract but do not "
                "guarantee identical random trajectories or floating-point reductions"
            ),
            "optimization": {
                "epochs": cfg.epochs,
                "backbone_learning_rate": cfg.learning_rate,
                "head_learning_rate": cfg.head_learning_rate,
                "schedule": "base_cosine_then_refinement_cosine",
                "refine_learning_rate": cfg.refine_learning_rate,
                "refine_head_learning_rate": cfg.refine_head_learning_rate,
                "refine_eta_min": cfg.refine_eta_min,
                "eta_min": 0.1 * cfg.learning_rate,
            },
            "training_objective": {
                "pose": "continuous_symmetry_aware_pose_and_size",
                "iou": "mean_iou_proxy",
                "irls_iterations": cfg.irls_iterations,
            },
        },
    }


def make_training_loader(cfg, distributed, rank, world_size, *, refining):
    return get_data_loaders(
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
        camera_real_ratio=cfg.refine_camera_real_ratio if refining else cfg.camera_real_ratio,
    )


def main() -> None:
    cfg = arguments()
    validate_auxiliary_config(cfg)
    os.environ["INFLUPOSE_FPS_BACKEND"] = cfg.fps_backend
    if cfg.num_points != 1024:
        raise ValueError("Stage2 locks train/test sampling to 1024 points")
    if cfg.epochs <= 0 or cfg.max_train_batches == 0:
        raise ValueError("training length is invalid")
    if min(cfg.learning_rate, cfg.head_learning_rate) <= 0.0:
        raise ValueError("both learning rates must be positive")
    if not 0 < cfg.base_epochs < cfg.epochs:
        raise ValueError("base_epochs must be between zero and the total epoch count")
    if not cfg.allow_noncanonical_schedule and (
        cfg.epochs != 50 or cfg.base_epochs != 30 or cfg.max_train_batches >= 0
        or cfg.seed != 260907
    ):
        raise ValueError("the default recipe requires 30 base + 20 refinement epochs, all batches and seed 260907")
    for name in REFINEMENT_DEFAULTS:
        if not math.isfinite(getattr(cfg, name)) or getattr(cfg, name) < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if min(cfg.refine_learning_rate, cfg.refine_head_learning_rate, cfg.refine_eta_min,
           cfg.refine_camera_real_ratio) <= 0:
        raise ValueError("refinement learning rates and sampling ratios must be positive")
    if not 0.0 <= cfg.observability_fraction <= 1.0:
        raise ValueError("observability_fraction must lie in [0,1]")
    if not 0.0 <= cfg.teacher_label_blend <= 1.0:
        raise ValueError("teacher_label_blend must lie in [0,1]")
    if cfg.camera_real_ratio <= 0.0:
        raise ValueError("data ratio must be positive")
    if cfg.irls_iterations != 3 and not getattr(
        cfg, "allow_noncanonical_solver_depth", False
    ):
        raise ValueError("the restored G10F contract requires exactly three IRLS steps")
    if cfg.irls_iterations < 0:
        raise ValueError("IRLS iterations must be non-negative")
    if min(
        cfg.irls_damping,
        cfg.max_rotation_step_degrees,
        cfg.max_translation_step,
        cfg.max_log_size_step,
    ) <= 0.0:
        raise ValueError("IRLS damping and update limits must be positive")
    if not 0.0 <= cfg.outlier_ratio < 1.0 or cfg.outlier_extent <= 0.0:
        raise ValueError("outlier corruption is invalid")
    if cfg.normals_k < 3:
        raise ValueError("normals_k must be at least 3")
    with open(cfg.observability_audit, "r", encoding="utf-8") as stream:
        observability_audit = json.load(stream)
    if observability_audit.get("passed") is not True:
        raise ValueError("Stage1 observability audit did not pass")
    expected_teacher_sha = observability_audit.get("checkpoint_sha256")
    if expected_teacher_sha and expected_teacher_sha != file_sha256(cfg.stage1_checkpoint):
        raise ValueError("observability audit is bound to another Stage1 checkpoint")

    distributed, rank, local_rank, world_size = configure_distributed(cfg)
    execution_protocol = record_runtime_execution_layout(
        cfg, world_size=world_size, distributed=distributed
    )
    is_main = rank == 0
    seed_everything(cfg.seed + rank)
    if cfg.device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
    os.makedirs(cfg.output_dir, exist_ok=True)
    teacher, stage1_state = load_stage1(cfg.stage1_checkpoint, cfg.device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    stage1_sha = file_sha256(cfg.stage1_checkpoint)

    bare_model = Stage2PoseInformationField(
        token_width=cfg.token_width,
        anchor_count=cfg.anchor_count,
        relation_layers=cfg.relation_layers,
        heads=cfg.heads,
        dropout=cfg.dropout,
        canonical_range=cfg.canonical_range,
        irls_iterations=cfg.irls_iterations,
        irls_damping=cfg.irls_damping,
        max_rotation_step_degrees=cfg.max_rotation_step_degrees,
        max_translation_step=cfg.max_translation_step,
        max_log_size_step=cfg.max_log_size_step,
    ).to(cfg.device)
    class_audit = class_prior_audit(bare_model)
    optimizer = make_optimizer(bare_model, cfg)
    scheduler = make_stage2_scheduler(optimizer, cfg)
    ema = ExponentialMovingAverage(bare_model.parameters(), decay=cfg.ema_decay)
    start_epoch = 0
    global_step = 0
    resume_metadata = None
    reference_state = None
    if cfg.resume:
        state = torch.load(cfg.resume, map_location=cfg.device)
        checkpoint_execution_layout(
            state, required_layout=str(execution_protocol["execution_layout"])
        )
        if state.get("format") != CHECKPOINT_FORMAT:
            raise ValueError("resume checkpoint is not Stage2")
        if state.get("training_protocol") != TRAINING_PROTOCOL:
            raise ValueError("resume checkpoint uses a different training protocol")
        if state.get("architecture", {}) != bare_model.architecture():
            raise ValueError("resume architecture differs")
        source_epochs = int(state.get("training_config", {}).get("epochs", -1))
        if source_epochs != cfg.epochs:
            raise ValueError("resume must preserve the original epoch horizon")
        source_config = state.get("training_config", {})
        require_same_training_dataset(
            source_config, cfg, context="resume checkpoint"
        )
        if state.get("stage1_checkpoint_sha256") != stage1_sha:
            raise ValueError("resume checkpoint used another Stage1 teacher")
        for key in (
            "seed",
            "max_train_batches",
            "batch_size",
            "fps_backend",
            "camera_real_ratio",
            "learning_rate",
            "head_learning_rate",
            "weight_decay",
            "ema_decay",
            "base_epochs",
            "refine_learning_rate", "refine_head_learning_rate", "refine_eta_min",
            "refine_camera_real_ratio",
            *REFINEMENT_DEFAULTS,
            *AUXILIARY_DEFAULTS,
        ):
            if source_config.get(key) != getattr(cfg, key):
                raise ValueError(f"resume changed the canonical G10F setting: {key}")
        bare_model.load_state_dict(state["model_state_dict"], strict=True)
        refining_checkpoint = int(state["epoch"]) >= cfg.base_epochs
        expected_phase = "refinement" if refining_checkpoint else "base"
        if state.get("training_phase") != expected_phase:
            raise ValueError("resume checkpoint has an inconsistent training phase")
        optimizer = make_optimizer(bare_model, cfg, refining=refining_checkpoint)
        scheduler = make_stage2_scheduler(optimizer, cfg, refining=refining_checkpoint)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        ema.load_state_dict(state["ema_state_dict"])
        start_epoch = int(state["epoch"]) + 1
        global_step = int(state.get("global_step", 0))
        resume_metadata = {
            "path": os.path.abspath(cfg.resume),
            "sha256": file_sha256(cfg.resume),
            "source_epoch": start_epoch,
            "source_global_step": global_step,
            "optimizer_restored": True,
            "scheduler_restored": True,
            "ema_restored": True,
        }
        if state.get("reference_policy") != "frozen_base_endpoint_ema":
            raise ValueError("resume checkpoint uses a different reference policy")
        reference_state = state.get("reference_model_state_dict")
        if refining_checkpoint and (not reference_state or state.get("reference_epoch") != cfg.base_epochs):
            raise ValueError("refinement checkpoint lacks its frozen base endpoint reference")
        if not refining_checkpoint and reference_state is not None:
            raise ValueError("base checkpoint must not contain a non-regression reference")
    endpoint_audit = contract_audit(bare_model, cfg.device)
    model = (
        DistributedDataParallel(
            bare_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
        if distributed
        else bare_model
    )
    if not cfg.resume:
        # Initialize EMA after DDP broadcasts rank-zero model parameters.
        ema = ExponentialMovingAverage(bare_model.parameters(), decay=cfg.ema_decay)
    reference = frozen_reference(bare_model, reference_state) if reference_state else None
    loader = make_training_loader(cfg, distributed, rank, world_size, refining=start_epoch >= cfg.base_epochs)
    if is_main:
        print(json.dumps({
            "run": "stage2-observability-guided-pose-information-field",
            "checkpoint_format": CHECKPOINT_FORMAT,
            "stage1_teacher_epoch": int(stage1_state.get("epoch", -1)) + 1,
            "world_size": world_size,
            "global_batch_size": cfg.batch_size * world_size,
            "execution_protocol": execution_protocol,
            "max_train_batches": cfg.max_train_batches,
            "pointcloud_only_loader": True,
            "fps_backend": cfg.fps_backend,
            "rgb_enabled": False,
            "class_prior_audit": class_audit,
            "contract_audit": endpoint_audit,
            "full_dataset_training": cfg.max_train_batches < 0,
            "optimization_protocol": TRAINING_PROTOCOL,
            "base_epochs": cfg.base_epochs,
            "refinement_epochs": cfg.epochs - cfg.base_epochs,
            "final_solver": solver_contract(cfg.irls_iterations),
            "ema_warmup": True,
            "resume": resume_metadata,
        }, sort_keys=True), flush=True)
    if distributed:
        dist.barrier()

    history_path = os.path.join(cfg.output_dir, "history.jsonl")
    use_amp = cfg.amp == "bfloat16"
    for epoch in range(start_epoch, cfg.epochs):
        refining = epoch >= cfg.base_epochs
        if epoch == cfg.base_epochs:
            optimizer, scheduler, ema, reference, reference_state = begin_refinement(bare_model, ema, cfg)
            loader = make_training_loader(cfg, distributed, rank, world_size, refining=True)
        if refining and reference is None:
            raise RuntimeError("refinement requires the frozen base endpoint reference")
        if distributed and hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)
        epoch_learning_rates = [
            float(group["lr"]) for group in optimizer.param_groups
        ]
        model.train()
        totals: dict[str, float] = {}
        examples = 0
        skipped = 0
        if cfg.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(cfg.device)
        progress = tqdm(loader, desc=f"Stage2 {epoch + 1}/{cfg.epochs}", disable=not is_main)
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
            clean_points = batch["pts"].float()
            points, input_inlier = inject_outliers(
                clean_points, cfg.outlier_ratio, cfg.outlier_extent
            )
            center = batch["pts_center"].float()
            target_pose = batch["gt_pose"].float()
            target_size = batch["gt_size"].float()
            class_ids = batch["id"].long()
            handles = batch["handle_visibility"].long()
            symmetric = continuous_y_symmetry_mask(class_ids, handles)
            normals, observation = estimate_camera_observation_features(
                points,
                center,
                k=cfg.normals_k,
            )
            normals = normals.detach()
            observation = observation.detach()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                output, view_loss, view_count, view_data = forward_training_views(
                    model, points, normals, observation, center, symmetric,
                    class_ids, handles, cfg, batch_index=batch_index, enabled=refining,
                )
                clean_q = inverse_pose_coordinates(clean_points, target_pose, target_size)
                anchor_target_q = gather_rows(clean_q, output["anchor_indices"])
                anchor_inlier = torch.gather(input_inlier, 1, output["anchor_indices"])
                canonical_point_error = coordinate_error(
                    output["canonical"], anchor_target_q, symmetric
                )
                canonical = masked_mean(canonical_point_error, anchor_inlier)
                orientation_canonical = canonical.new_zeros(())
                if refining:
                    oriented_error = F.smooth_l1_loss(
                        output["canonical"], anchor_target_q, beta=0.03, reduction="none"
                    ).mean(dim=-1)
                    orientation_canonical = masked_mean(oriented_error, anchor_inlier & (~symmetric[:, None]))
                relation = relation_error(
                    output["canonical"],
                    anchor_target_q,
                    output["size"],
                    target_size,
                    anchor_inlier,
                )
                inlier = F.binary_cross_entropy_with_logits(
                    output["inlier_logits"], anchor_inlier.to(output["inlier_logits"].dtype)
                )
                target_rotation = pose6d_to_matrix(target_pose[:, :6])
                predicted_in_target_camera = torch.einsum(
                    "bij,bkj->bki",
                    target_rotation,
                    output["canonical"] * target_size[:, None],
                ) + target_pose[:, None, 6:]
                camera_residual = predicted_in_target_camera - output["anchor_xyz"]
                precision = output["axis_precision"].clamp(0.05, 50.0)
                nll_point = 0.5 * (
                    camera_residual.square() * precision - torch.log(precision)
                ).mean(dim=-1)
                uncertainty = masked_mean(nll_point, anchor_inlier)

                information_target = analytic_correspondence_information(
                    anchor_target_q, target_size, symmetric
                )
            measured_count = 0
            if (
                cfg.observability_fraction > 0.0
                and batch_index % cfg.observability_interval == 0
            ):
                measured_count = max(
                    1, int(round(points.shape[0] * cfg.observability_fraction))
                )
                full_points = batch["full_cloud"][:measured_count].float()
                full_normals = estimate_point_normals(
                    full_points,
                    full_points.mean(dim=1),
                    k=cfg.normals_k,
                ).detach()
                with torch.no_grad(), torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=use_amp
                ):
                    measured = regional_stage1_observability(
                        teacher,
                        full_points,
                        full_normals,
                        center[:measured_count],
                        class_ids[:measured_count],
                        handles[:measured_count],
                        region_count=cfg.observability_regions,
                        removal_fraction=cfg.observability_removal_fraction,
                        repeat_offset=(global_step // max(1, cfg.observability_interval)) % 3,
                        teacher_chunk=cfg.observability_teacher_chunk,
                    )
                    transferred = transfer_region_information(
                        output["anchor_xyz"][:measured_count].detach(),
                        measured["region_centers"],
                        measured["normalized_impact"],
                    ).to(information_target.dtype)
                information_target = information_target.clone()
                information_target[:measured_count] = (
                    (1.0 - cfg.teacher_label_blend) * information_target[:measured_count]
                    + cfg.teacher_label_blend * transferred
                )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                info_regression, info_kl, info_mass = information_losses(
                    output["information"],
                    information_target,
                    anchor_inlier,
                    symmetric,
                    cfg.information_temperature,
                )
                pose_task, pose_stats = pose_box_loss(
                    output,
                    target_pose,
                    target_size,
                    class_ids,
                    handles,
                    cfg.size_loss_weight,
                )
                box_iou = symmetry_aware_box_iou_proxy(
                    output["pose"],
                    output["size"],
                    target_pose,
                    target_size,
                    class_ids,
                    handles,
                )
                iou_loss = 1.0 - box_iou.mean()
                robust_pose_loss = box_iou.new_zeros(())
                if view_data is not None:
                    secondary, indices = view_data
                    robust_pose, _ = pose_box_loss(
                        secondary, target_pose[indices], target_size[indices],
                        class_ids[indices], handles[indices], cfg.size_loss_weight,
                    )
                    robust_pose_loss = cfg.refine_view_pose_weight * cfg.pose_weight * robust_pose
                initial_size_loss = F.smooth_l1_loss(
                    torch.log(output["network_initial_size"].clamp_min(1e-5)),
                    torch.log(target_size.clamp_min(1e-5)),
                    beta=0.10,
                )
                reference_pose = points.new_zeros(())
                reference_iou = points.new_zeros(())
                reference_stats = {}
                if reference is not None:
                    with torch.no_grad():
                        reference_output = reference(
                            points, normals, observation,
                            alignment_center=center, continuous_symmetry=symmetric,
                        )
                    reference_pose, reference_iou, reference_stats = (
                        reference_non_regression_loss(
                            output, reference_output, target_pose, target_size,
                            class_ids, handles, student_iou=box_iou,
                            rotation_margin_degrees=cfg.reference_rotation_margin_degrees,
                            translation_margin_cm=cfg.reference_translation_margin_cm,
                            iou_margin=cfg.reference_iou_margin,
                        )
                    )
                loss = (
                    cfg.pose_weight * pose_task
                    + cfg.iou_weight * iou_loss
                    + cfg.canonical_weight * canonical
                    + cfg.relation_weight * relation
                    + cfg.inlier_weight * inlier
                    + cfg.uncertainty_weight * uncertainty
                    + cfg.information_regression_weight * info_regression
                    + cfg.information_kl_weight * info_kl
                    + cfg.information_mass_weight * info_mass
                    + cfg.initial_size_weight * initial_size_loss
                    + cfg.refine_orientation_canonical_weight * orientation_canonical
                    + robust_pose_loss
                    + cfg.view_consistency_weight * view_loss
                    + cfg.reference_pose_weight * reference_pose
                    + cfg.reference_iou_weight * reference_iou
                )
            if not torch.isfinite(loss):
                skipped += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(bare_model.parameters(), cfg.grad_clip)
            if not torch.isfinite(gradient_norm):
                skipped += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.step()
            ema.update(bare_model.parameters())
            global_step += 1
            count = points.shape[0]
            examples += count
            statistics = {
                "loss": loss.detach(),
                "pose": pose_task.detach(),
                "iou": iou_loss.detach(),
                "mean_iou": box_iou.mean().detach(),
                "canonical": canonical.detach(),
                "relation": relation.detach(),
                "inlier": inlier.detach(),
                "uncertainty": uncertainty.detach(),
                "information_regression": info_regression.detach(),
                "information_kl": info_kl.detach(),
                "information_mass": info_mass.detach(),
                "initial_size": initial_size_loss.detach(),
                "view_consistency": view_loss.detach(),
                "robust_view_pose": robust_pose_loss.detach(),
                "orientation_canonical": orientation_canonical.detach(),
                "view_examples": points.new_tensor(float(view_count)),
                "reference_pose": reference_pose.detach(),
                "reference_iou": reference_iou.detach(),
                **reference_stats,
                "rotation_deg": pose_stats["rotation_deg"],
                "translation_cm": pose_stats["translation_cm"],
                "size_relative_percent": pose_stats["size_relative_percent"],
                "solver_gate": output["solver_gate"].mean().detach(),
                "solver_condition_log10": torch.log10(
                    output["solver_condition"].clamp_min(1.0)
                ).mean().detach(),
                "measured_observability_examples": torch.tensor(float(measured_count), device=cfg.device),
            }
            for key, value in statistics.items():
                totals[key] = totals.get(key, 0.0) + float(value) * count
            if is_main and batch_index % 20 == 0:
                progress.set_postfix(
                    loss=f"{float(loss):.3f}",
                    rot=f"{float(pose_stats['rotation_deg']):.2f}",
                    iou=f"{float(box_iou.mean()):.3f}",
                    q=f"{float(canonical):.3f}",
                )

        scheduler.step()
        totals, examples = reduce_epoch_totals(
            totals, examples, device=cfg.device, distributed=distributed
        )
        if distributed:
            skipped_tensor = torch.tensor(skipped, device=cfg.device, dtype=torch.long)
            dist.all_reduce(skipped_tensor)
            skipped = int(skipped_tensor)
        peak_allocated_gib = 0.0
        peak_reserved_gib = 0.0
        if cfg.device.startswith("cuda"):
            peak_allocated = torch.tensor(
                float(torch.cuda.max_memory_allocated(cfg.device)),
                device=cfg.device,
                dtype=torch.float64,
            )
            peak_reserved = torch.tensor(
                float(torch.cuda.max_memory_reserved(cfg.device)),
                device=cfg.device,
                dtype=torch.float64,
            )
            if distributed:
                dist.all_reduce(peak_allocated, op=dist.ReduceOp.MAX)
                dist.all_reduce(peak_reserved, op=dist.ReduceOp.MAX)
            gibibyte = float(1024 ** 3)
            peak_allocated_gib = float(peak_allocated.item()) / gibibyte
            peak_reserved_gib = float(peak_reserved.item()) / gibibyte
        record = {
            "epoch": epoch,
            "training_phase": "refinement" if refining else "base",
            "backbone_learning_rate": epoch_learning_rates[0],
            "head_learning_rate": epoch_learning_rates[1],
            "next_backbone_learning_rate": float(optimizer.param_groups[0]["lr"]),
            "next_head_learning_rate": float(optimizer.param_groups[1]["lr"]),
            "global_step": global_step,
            "examples": examples,
            "peak_cuda_allocated_gib": peak_allocated_gib,
            "peak_cuda_reserved_gib": peak_reserved_gib,
            "skipped_nonfinite_steps": skipped,
            **{key: value / max(1, examples) for key, value in totals.items()},
        }
        if distributed:
            dist.barrier()
        if is_main:
            history_record = record
            with open(history_path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(history_record, sort_keys=True) + "\n")
            print(json.dumps(history_record, sort_keys=True), flush=True)
            payload = checkpoint_payload(
                bare_model,
                optimizer,
                scheduler,
                ema,
                epoch,
                global_step,
                cfg,
                stage1_sha,
                endpoint_audit,
                reference_state,
            )
            epoch_checkpoint_path = os.path.join(
                cfg.output_dir, f"epoch_{epoch + 1:03d}.pth"
            )
            latest_checkpoint_path = os.path.join(cfg.output_dir, "latest.pth")
            torch.save(payload, epoch_checkpoint_path)
            torch.save(payload, latest_checkpoint_path)
            if epoch + 1 == cfg.epochs:
                shutil.copy2(
                    epoch_checkpoint_path,
                    os.path.join(cfg.output_dir, "final.pth"),
                )

        if distributed:
            dist.barrier()

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
