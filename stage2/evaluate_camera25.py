"""Evaluate Stage2 on locally valid CAMERA25 frames with GT instance masks."""

from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from shared.dataset import NOCSDataSet, process_batch
from shared.point_features import estimate_camera_observation_features
from shared.pose_geometry import continuous_y_symmetry_mask, pose6d_to_matrix
from shared.real275_metrics import (
    compute_real275_3d_map,
    per_class_diagnostics,
    summary_from_metrics,
)
from shared.training import NO_MASK_DEFORM, NO_ZOOM, seed_everything
from stage2.evaluate_real275 import file_sha256, load_model


class CameraInstances(Dataset):
    def __init__(self, base: NOCSDataSet, specs: list[tuple[int, int, int]]) -> None:
        self.base = base
        self.specs = specs

    def __len__(self) -> int:
        return len(self.specs)

    def __getitem__(self, index: int) -> dict:
        frame_index, instance_index, point_seed = self.specs[index]
        sample = self.base[(frame_index, instance_index, point_seed)]
        if int(sample["instance_index"]) != instance_index:
            raise RuntimeError("CAMERA instance loader changed the forced instance")
        return sample


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", default="data/nocs")
    parser.add_argument("--checkpoint", default="checkpoints/stage2.pth")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--normals_k", type=int, default=16)
    parser.add_argument("--max_instances", type=int, default=0)
    parser.add_argument("--weights", choices=("auto", "ema", "raw"), default="auto")
    parser.add_argument(
        "--irls_iterations",
        type=int,
        default=None,
        help=(
            "Number of robust solver updates; omit to use the checkpoint setting."
        ),
    )
    parser.add_argument("--seed", type=int, default=260925)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _frame_result(label: dict) -> dict:
    return {
        "gt_class_ids": np.asarray(label["class_ids"], dtype=np.int32),
        "gt_bboxes": np.asarray(label["bboxes"]),
        "gt_RTs": np.asarray(label["poses"], dtype=np.float64),
        "gt_scales": np.asarray(label["size"], dtype=np.float64),
        "gt_handle_visibility": np.asarray(
            label.get("handle_visibility", np.ones(len(label["class_ids"]))),
            dtype=np.int32,
        ),
        "pred_class_ids": [],
        "pred_bboxes": [],
        "pred_scores": [],
        "pred_RTs": [],
        "pred_scales": [],
    }


def main() -> None:
    cfg = arguments()
    if cfg.max_instances < 0 or min(cfg.batch_size, cfg.num_workers + 1) <= 0:
        raise ValueError("invalid evaluation size")
    if cfg.irls_iterations is not None and cfg.irls_iterations < 0:
        raise ValueError("irls_iterations must be non-negative")
    seed_everything(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)
    model, state = load_model(cfg.checkpoint, cfg.device, cfg.weights)
    checkpoint_irls_iterations = int(model.irls_iterations)
    if cfg.irls_iterations is not None:
        model.irls_iterations = int(cfg.irls_iterations)
    base = NOCSDataSet(
        dynamic_zoom_in_params=NO_ZOOM,
        deform_2d_params=NO_MASK_DEFORM,
        source="CAMERA",
        mode="test",
        data_dir=cfg.data_path,
        n_pts=1024,
        img_size=256,
        pointcloud_only=True,
    )

    specs: list[tuple[int, int, int]] = []
    frames: dict[str, dict] = {}
    labels: dict[str, dict] = {}
    for frame_index, relative in enumerate(base.img_list):
        path = os.path.join(cfg.data_path, relative)
        with open(path + "_label.pkl", "rb") as stream:
            label = pickle.load(stream)
        labels[path] = label
        frames[path] = _frame_result(label)
        for instance_index in range(len(label["class_ids"])):
            specs.append((frame_index, instance_index, cfg.seed + len(specs)))
    total_instances = len(specs)
    if cfg.max_instances:
        specs = specs[: cfg.max_instances]
        retained_paths = {
            os.path.join(cfg.data_path, base.img_list[frame_index])
            for frame_index, _, _ in specs
        }
        frames = {path: value for path, value in frames.items() if path in retained_paths}
        labels = {path: value for path, value in labels.items() if path in retained_paths}

    loader = DataLoader(
        CameraInstances(base, specs),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
        pin_memory=True,
        drop_last=False,
    )
    predicted = 0
    use_amp = cfg.device.startswith("cuda")
    with torch.inference_mode():
        for raw in tqdm(loader, desc="Stage2 CAMERA25 GT-mask ceiling"):
            batch = process_batch(
                raw,
                device=cfg.device,
                pose_mode="rot_matrix",
                PTS_AUG_PARAMS=None,
            )
            points = batch["pts"].float()
            centers = points.mean(dim=1)
            class_ids = batch["id"].long()
            normals, observation = estimate_camera_observation_features(
                points, centers, k=cfg.normals_k
            )
            symmetric = continuous_y_symmetry_mask(
                class_ids, batch["handle_visibility"].long()
            )
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=use_amp
            ):
                output = model(
                    points,
                    normals,
                    observation,
                    alignment_center=centers,
                    continuous_symmetry=symmetric,
                )

            rotations = pose6d_to_matrix(output["pose"][:, :6]).cpu().numpy()
            translations = output["pose"][:, 6:].float().cpu().numpy()
            sizes = output["size"].float().cpu().numpy()
            paths = list(raw["path"])
            instance_indices = raw["instance_index"].cpu().numpy().astype(int)
            categories = raw["cat_id"].cpu().numpy().astype(int) + 1
            for offset, path in enumerate(paths):
                instance_index = int(instance_indices[offset])
                transform = np.eye(4, dtype=np.float64)
                transform[:3, :3] = rotations[offset]
                transform[:3, 3] = translations[offset]
                frames[path]["pred_class_ids"].append(categories[offset])
                frames[path]["pred_bboxes"].append(
                    np.asarray(labels[path]["bboxes"])[instance_index]
                )
                frames[path]["pred_scores"].append(1.0)
                frames[path]["pred_RTs"].append(transform)
                frames[path]["pred_scales"].append(sizes[offset])
                predicted += 1

    results: list[dict] = []
    for result in frames.values():
        result["pred_class_ids"] = np.asarray(result["pred_class_ids"], dtype=np.int32)
        result["pred_bboxes"] = np.asarray(result["pred_bboxes"])
        result["pred_scores"] = np.asarray(result["pred_scores"], dtype=np.float32)
        result["pred_RTs"] = np.asarray(result["pred_RTs"], dtype=np.float64).reshape(-1, 4, 4)
        result["pred_scales"] = np.asarray(result["pred_scales"], dtype=np.float64).reshape(-1, 3)
        results.append(result)

    metrics = compute_real275_3d_map(
        results, os.path.join(cfg.output_dir, "official_metrics_3d")
    )
    summary = summary_from_metrics(metrics)
    full = cfg.max_instances == 0
    summary.update(
        {
            "protocol": "CAMERA25_VALID_FRAME_GT_MASK_SYNTHETIC_CEILING",
            "dataset_frames": len(frames),
            "dataset_instances": total_instances,
            "predicted_instances": predicted,
            "full_local_valid_split": full,
            "official_camera25_images": 25000,
            "local_valid_preprocessed_images": 19220,
            "evaluated_images": len(frames),
            "missing_preprocessed_labels_from_official_count": 5780,
            "segmentation": "ground_truth_instance_mask_and_bbox",
            "checkpoint": os.path.abspath(cfg.checkpoint),
            "checkpoint_sha256": file_sha256(cfg.checkpoint),
            "evaluated_weights": state["resolved_evaluation_weights"],
            "checkpoint_irls_iterations": checkpoint_irls_iterations,
            "evaluation_irls_iterations": int(model.irls_iterations),
            "evaluation_batch_size": cfg.batch_size,
            "per_class_diagnostics": per_class_diagnostics(metrics),
        }
    )
    with open(
        os.path.join(cfg.output_dir, "evaluation_summary.json"),
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
