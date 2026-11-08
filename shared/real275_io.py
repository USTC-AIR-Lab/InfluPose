"""REAL275 detector-cache adapters shared by both formal stages."""

from __future__ import annotations

import copy
import os
import pickle
from collections import defaultdict

import numpy as np


def empty_prediction(cache: dict) -> dict:
    """Clone detector outputs and allocate one pose/box slot per detection."""

    output = copy.deepcopy(cache)
    for item in output.values():
        count = len(item["result"]["pred_class_ids"])
        item["result"]["pred_RTs"] = np.repeat(
            np.eye(4, dtype=np.float64)[None], count, axis=0
        )
        item["result"]["pred_scales"] = np.ones((count, 3), dtype=np.float64)
    return output


def records_from_detection_cache(
    cache: dict, expected_points: int
) -> tuple[dict[str, list], dict[str, int]]:
    """Extract valid detector partial clouds without reading RGB images."""

    records: dict[str, list] = defaultdict(list)
    valid = 0
    invalid = 0
    for image_path, item in cache.items():
        result = item["result"]
        visible_list = item.get("valid_pts", [])
        categories = item.get("cat_id", [])
        for local_index, prediction_index in enumerate(item.get("valid_inst", [])):
            prediction_index = int(prediction_index)
            valid += 1
            if (
                local_index >= len(visible_list)
                or local_index >= len(categories)
                or not 0 <= prediction_index < len(result["pred_class_ids"])
            ):
                invalid += 1
                continue
            points = np.asarray(visible_list[local_index], dtype=np.float32)
            if points.shape != (expected_points, 3) or not np.isfinite(points).all():
                invalid += 1
                continue
            class_id = int(result["pred_class_ids"][prediction_index]) - 1
            if class_id != int(categories[local_index]) or not 0 <= class_id < 6:
                invalid += 1
                continue
            anchor = points.mean(axis=0, dtype=np.float32)
            records["image"].append(image_path)
            records["prediction_index"].append(prediction_index)
            records["class_id"].append(class_id)
            records["partial_relative"].append(points - anchor[None])
            records["alignment_center"].append(anchor)
    return records, {
        "valid_detections": valid,
        "completed_inputs": len(records["image"]),
        "invalid_detections": invalid,
    }


def _relative_real_path(image_path: str) -> str:
    if image_path.startswith("Real/"):
        return image_path
    marker = "/Real/"
    if marker not in image_path:
        raise ValueError(f"not a REAL275 image path: {image_path}")
    return image_path[image_path.index(marker) + 1 :]


def _bbox_iou(first: np.ndarray, second: np.ndarray) -> float:
    top = max(float(first[0]), float(second[0]))
    left = max(float(first[1]), float(second[1]))
    bottom = min(float(first[2]), float(second[2]))
    right = min(float(first[3]), float(second[3]))
    intersection = max(0.0, bottom - top) * max(0.0, right - left)
    first_area = max(0.0, float(first[2] - first[0])) * max(
        0.0, float(first[3] - first[1])
    )
    second_area = max(0.0, float(second[2] - second[0])) * max(
        0.0, float(second[3] - second[1])
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _visible_cloud_center(item: dict, local_index: int) -> np.ndarray:
    visible = item.get("valid_pts")
    if not isinstance(visible, (list, tuple)) or local_index >= len(visible):
        raise ValueError("detector cache has no visible cloud for valid instance")
    points = np.asarray(visible[local_index], dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError("detector visible cloud must have shape [N>=3,3]")
    if not np.isfinite(points).all():
        raise ValueError("detector visible cloud contains non-finite values")
    return points.mean(axis=0, dtype=np.float32)


def oracle_complete_cloud_records(
    cache: dict, data_path: str
) -> tuple[dict[str, list], dict[str, float | int]]:
    """Build Stage1 complete-cloud inputs from the REAL275 annotations."""

    with open(os.path.join(data_path, "obj_models", "real_test.pkl"), "rb") as stream:
        models = pickle.load(stream)
    records: dict[str, list] = defaultdict(list)
    label_cache: dict[str, dict] = {}
    ious: list[float] = []
    unmatched = 0
    invalid_visible = 0
    valid = 0
    for image_path, item in cache.items():
        label_path = os.path.join(
            data_path, _relative_real_path(image_path) + "_label.pkl"
        )
        if label_path not in label_cache:
            with open(label_path, "rb") as stream:
                label_cache[label_path] = pickle.load(stream)
        labels = label_cache[label_path]
        result = item["result"]
        for local_index, prediction_index in enumerate(item["valid_inst"]):
            prediction_index = int(prediction_index)
            valid += 1
            try:
                anchor = _visible_cloud_center(item, local_index)
            except ValueError:
                invalid_visible += 1
                unmatched += 1
                continue
            if prediction_index >= len(result["pred_class_ids"]):
                unmatched += 1
                continue
            one_based_class = int(result["pred_class_ids"][prediction_index])
            candidates = np.flatnonzero(labels["class_ids"] == one_based_class)
            if not len(candidates):
                unmatched += 1
                continue
            predicted_box = np.asarray(result["pred_bboxes"][prediction_index])
            candidate_ious = [
                _bbox_iou(predicted_box, labels["bboxes"][index])
                for index in candidates
            ]
            chosen = int(candidates[int(np.argmax(candidate_ious))])
            match_iou = float(max(candidate_ious))
            if match_iou < 0.5:
                unmatched += 1
                continue
            model_name = labels["model_list"][chosen]
            full_cloud = (
                np.asarray(labels["rotations"][chosen], dtype=np.float32)
                @ (
                    np.asarray(models[model_name], dtype=np.float32)
                    * float(labels["scales"][chosen])
                ).T
            ).T + np.asarray(labels["translations"][chosen], dtype=np.float32)[None]
            records["image"].append(image_path)
            records["prediction_index"].append(prediction_index)
            records["class_id"].append(one_based_class - 1)
            records["points"].append(full_cloud)
            records["alignment_center"].append(anchor)
            ious.append(match_iou)
    return records, {
        "valid_detections": valid,
        "matched_detections": len(ious),
        "unmatched_detections": unmatched,
        "invalid_visible_cloud": invalid_visible,
        "mean_match_iou": float(np.mean(ious)) if ious else 0.0,
        "minimum_match_iou": float(np.min(ious)) if ious else 0.0,
    }
