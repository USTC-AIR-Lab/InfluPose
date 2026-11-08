"""REAL275 3D-box and pose AP evaluation for one-pose predictions.

* 3D IoU AP is matched with predicted 3D boxes at the requested thresholds;
* pose AP is matched only after a genuine 3D IoU >= 0.1 match;
* the input contains one predicted sRT and one predicted metric box size per
  detector instance.

The geometry, symmetry handling, and AP interpolation are intentionally
reused from the repository's NOCS implementation.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass

import numpy as np

from shared.nocs_metrics import (
    compute_IoU_matches,
    compute_RT_matches,
    compute_RT_overlaps,
    compute_ap_and_acc,
)


SYNSET_NAMES = ("BG", "bottle", "bowl", "camera", "can", "laptop", "mug")


@dataclass(frozen=True)
class Real275MetricResult:
    """Dense protocol outputs, using the NOCS ``BG + six classes + mean`` layout."""

    iou_aps: np.ndarray
    pose_aps: np.ndarray
    iou_acc: np.ndarray
    pose_acc: np.ndarray
    iou_thresholds: tuple[float, ...]
    degree_thresholds: tuple[int, ...]
    shift_thresholds: tuple[float, ...]


def _empty_matches(
    iou_count: int, degree_count: int, shift_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros((iou_count, 0), dtype=np.float64),
        np.zeros((iou_count, 0), dtype=np.float64),
        np.zeros((degree_count, shift_count, 0), dtype=np.float64),
        np.zeros((degree_count, shift_count, 0), dtype=np.float64),
    )


def _safe_ap(pred_matches: np.ndarray, pred_scores: np.ndarray, gt_matches: np.ndarray) -> tuple[float, float]:
    """Match the official AP implementation while handling an empty class safely."""

    if pred_matches.size == 0 or gt_matches.size == 0:
        return 0.0, 0.0
    return compute_ap_and_acc(pred_matches, pred_scores, gt_matches)


def compute_real275_3d_map(
    prediction_results: list[dict],
    output_dir: str,
    *,
    iou_thresholds: tuple[float, ...] = (0.1, 0.5, 0.75),
    degree_thresholds: tuple[int, ...] = tuple(range(46)),
    shift_thresholds: tuple[float, ...] = tuple(index / 2 for index in range(21)),
    pose_match_iou: float = 0.1,
) -> Real275MetricResult:
    """Compute the ComPose-compatible REAL275 AP protocol.

    ``prediction_results`` must retain detector ``pred_class_ids``,
    ``pred_scores`` and prediction ordering.  It replaces only ``pred_RTs``
    and ``pred_scales`` with retained-model outputs.  Ground-truth sRT can
    contain the standard NOCS uniform scale while the predicted sRT contains
    a pure rotation; this is valid because metric dimensions are supplied
    separately in ``pred_scales``.
    """

    if pose_match_iou not in iou_thresholds:
        raise ValueError("pose_match_iou must be included in iou_thresholds")

    classes = len(SYNSET_NAMES)
    iou_count = len(iou_thresholds)
    degree_with_terminal = tuple(degree_thresholds) + (360,)
    shift_with_terminal = tuple(shift_thresholds) + (100.0,)
    degree_count = len(degree_with_terminal)
    shift_count = len(shift_with_terminal)

    iou_prediction_matches: list[list[np.ndarray]] = [[] for _ in range(classes)]
    iou_prediction_scores: list[list[np.ndarray]] = [[] for _ in range(classes)]
    iou_ground_truth_matches: list[list[np.ndarray]] = [[] for _ in range(classes)]
    pose_prediction_matches: list[list[np.ndarray]] = [[] for _ in range(classes)]
    pose_prediction_scores: list[list[np.ndarray]] = [[] for _ in range(classes)]
    pose_ground_truth_matches: list[list[np.ndarray]] = [[] for _ in range(classes)]

    pose_iou_index = iou_thresholds.index(pose_match_iou)
    for result in prediction_results:
        gt_class_ids = np.asarray(result["gt_class_ids"], dtype=np.int32)
        gt_srt = np.asarray(result["gt_RTs"], dtype=np.float64)
        gt_size = np.asarray(result["gt_scales"], dtype=np.float64)
        gt_handle_visibility = np.asarray(result.get("gt_handle_visibility", np.ones_like(gt_class_ids)))
        pred_class_ids = np.asarray(result["pred_class_ids"], dtype=np.int32)
        pred_srt = np.asarray(result["pred_RTs"], dtype=np.float64)
        pred_size = np.asarray(result["pred_scales"], dtype=np.float64)
        pred_scores = np.asarray(result["pred_scores"], dtype=np.float64)

        if pred_srt.shape != (len(pred_class_ids), 4, 4):
            raise ValueError("pred_RTs must align as [num_predictions, 4, 4]")
        if pred_size.shape != (len(pred_class_ids), 3):
            raise ValueError("pred_scales must align as [num_predictions, 3]")

        for class_id in range(1, classes):
            gt_mask = gt_class_ids == class_id
            pred_mask = pred_class_ids == class_id
            class_gt_ids = gt_class_ids[gt_mask]
            class_gt_srt = gt_srt[gt_mask]
            class_gt_size = gt_size[gt_mask]
            class_pred_ids = pred_class_ids[pred_mask]
            class_pred_srt = pred_srt[pred_mask]
            class_pred_size = pred_size[pred_mask]
            class_pred_scores = pred_scores[pred_mask]
            if SYNSET_NAMES[class_id] == "mug":
                class_handle = gt_handle_visibility[gt_mask]
            else:
                class_handle = np.ones_like(class_gt_ids)

            gt_iou_match, pred_iou_match, _, score_order = compute_IoU_matches(
                class_gt_ids,
                class_gt_srt,
                class_gt_size,
                class_handle,
                class_pred_ids,
                class_pred_srt,
                class_pred_size,
                class_pred_scores,
                SYNSET_NAMES,
                iou_thresholds,
            )
            # ``compute_IoU_matches`` returns match indices in descending-score
            # prediction order, so all prediction tensors must be reordered.
            if len(score_order):
                class_pred_ids = class_pred_ids[score_order]
                class_pred_srt = class_pred_srt[score_order]
                class_pred_scores = class_pred_scores[score_order]

            iou_prediction_matches[class_id].append(pred_iou_match)
            iou_prediction_scores[class_id].append(
                np.tile(class_pred_scores, (iou_count, 1))
            )
            iou_ground_truth_matches[class_id].append(gt_iou_match)

            matched_predictions = pred_iou_match[pose_iou_index] > -1
            matched_ground_truth = gt_iou_match[pose_iou_index] > -1
            pose_pred_ids = class_pred_ids[matched_predictions]
            pose_pred_srt = class_pred_srt[matched_predictions]
            pose_pred_scores = class_pred_scores[matched_predictions]
            pose_gt_ids = class_gt_ids[matched_ground_truth]
            pose_gt_srt = class_gt_srt[matched_ground_truth]
            pose_gt_handle = class_handle[matched_ground_truth]
            overlaps = compute_RT_overlaps(
                pose_gt_ids,
                pose_gt_srt,
                pose_gt_handle,
                pose_pred_ids,
                pose_pred_srt,
                SYNSET_NAMES,
            )
            gt_pose_match, pred_pose_match = compute_RT_matches(
                overlaps,
                pose_pred_ids,
                pose_gt_ids,
                degree_with_terminal,
                shift_with_terminal,
            )
            pose_prediction_matches[class_id].append(pred_pose_match)
            pose_prediction_scores[class_id].append(
                np.broadcast_to(
                    pose_pred_scores[None, None, :],
                    (degree_count, shift_count, len(pose_pred_scores)),
                ).copy()
            )
            pose_ground_truth_matches[class_id].append(gt_pose_match)

    iou_aps = np.zeros((classes + 1, iou_count), dtype=np.float64)
    iou_acc = np.zeros_like(iou_aps)
    pose_aps = np.zeros((classes + 1, degree_count, shift_count), dtype=np.float64)
    pose_acc = np.zeros_like(pose_aps)
    for class_id in range(1, classes):
        if iou_prediction_matches[class_id]:
            pred_iou = np.concatenate(iou_prediction_matches[class_id], axis=1)
            pred_iou_scores = np.concatenate(iou_prediction_scores[class_id], axis=1)
            gt_iou = np.concatenate(iou_ground_truth_matches[class_id], axis=1)
            pred_pose = np.concatenate(pose_prediction_matches[class_id], axis=2)
            pred_pose_scores = np.concatenate(pose_prediction_scores[class_id], axis=2)
            gt_pose = np.concatenate(pose_ground_truth_matches[class_id], axis=2)
        else:
            pred_iou, pred_iou_scores, pred_pose, gt_pose = _empty_matches(
                iou_count, degree_count, shift_count
            )
            gt_iou = np.zeros((iou_count, 0), dtype=np.float64)
            pred_pose_scores = np.zeros((degree_count, shift_count, 0), dtype=np.float64)

        for index in range(iou_count):
            iou_aps[class_id, index], iou_acc[class_id, index] = _safe_ap(
                pred_iou[index], pred_iou_scores[index], gt_iou[index]
            )
        for degree_index in range(degree_count):
            for shift_index in range(shift_count):
                pose_aps[class_id, degree_index, shift_index], pose_acc[
                    class_id, degree_index, shift_index
                ] = _safe_ap(
                    pred_pose[degree_index, shift_index],
                    pred_pose_scores[degree_index, shift_index],
                    gt_pose[degree_index, shift_index],
                )

    iou_aps[-1] = np.mean(iou_aps[1:classes], axis=0)
    iou_acc[-1] = np.mean(iou_acc[1:classes], axis=0)
    pose_aps[-1] = np.mean(pose_aps[1:classes], axis=0)
    pose_acc[-1] = np.mean(pose_acc[1:classes], axis=0)
    metrics = Real275MetricResult(
        iou_aps=iou_aps,
        pose_aps=pose_aps,
        iou_acc=iou_acc,
        pose_acc=pose_acc,
        iou_thresholds=tuple(iou_thresholds),
        degree_thresholds=tuple(degree_with_terminal),
        shift_thresholds=tuple(shift_with_terminal),
    )
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "mAP_Acc_3d_protocol.pkl"), "wb") as stream:
        pickle.dump(
            {
                "iou_thres_list": metrics.iou_thresholds,
                "degree_thres_list": metrics.degree_thresholds,
                "shift_thres_list": metrics.shift_thresholds,
                "iou_aps": metrics.iou_aps,
                "pose_aps": metrics.pose_aps,
                "iou_acc": metrics.iou_acc,
                "pose_acc": metrics.pose_acc,
                "protocol": "REAL275_NOCS_3D_IOU_AND_POSE_AP",
            },
            stream,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    return metrics


def summary_from_metrics(metrics: Real275MetricResult) -> dict[str, float]:
    """Return the six table entries in the archived ComPose table order."""

    iou_50 = metrics.iou_thresholds.index(0.5)
    iou_75 = metrics.iou_thresholds.index(0.75)
    degree_5 = metrics.degree_thresholds.index(5)
    degree_10 = metrics.degree_thresholds.index(10)
    shift_2 = metrics.shift_thresholds.index(2.0)
    shift_5 = metrics.shift_thresholds.index(5.0)
    return {
        "3d_iou_50": float(metrics.iou_aps[-1, iou_50] * 100.0),
        "3d_iou_75": float(metrics.iou_aps[-1, iou_75] * 100.0),
        "5deg_2cm": float(metrics.pose_aps[-1, degree_5, shift_2] * 100.0),
        "5deg_5cm": float(metrics.pose_aps[-1, degree_5, shift_5] * 100.0),
        "10deg_2cm": float(metrics.pose_aps[-1, degree_10, shift_2] * 100.0),
        "10deg_5cm": float(metrics.pose_aps[-1, degree_10, shift_5] * 100.0),
    }


def per_class_diagnostics(metrics: Real275MetricResult) -> dict[str, dict[str, float]]:
    """Expose the rotation-vs-translation bottleneck hidden by mean AP."""

    iou_50 = metrics.iou_thresholds.index(0.5)
    iou_75 = metrics.iou_thresholds.index(0.75)
    degree_5 = metrics.degree_thresholds.index(5)
    degree_10 = metrics.degree_thresholds.index(10)
    rotation_free = metrics.degree_thresholds.index(360)
    shift_2 = metrics.shift_thresholds.index(2.0)
    shift_5 = metrics.shift_thresholds.index(5.0)
    translation_free = metrics.shift_thresholds.index(100.0)
    output: dict[str, dict[str, float]] = {}
    for class_id, name in enumerate(SYNSET_NAMES[1:], start=1):
        output[name] = {
            "3d_iou_50": float(metrics.iou_aps[class_id, iou_50] * 100.0),
            "3d_iou_75": float(metrics.iou_aps[class_id, iou_75] * 100.0),
            "5deg_2cm": float(
                metrics.pose_aps[class_id, degree_5, shift_2] * 100.0
            ),
            "5deg_5cm": float(
                metrics.pose_aps[class_id, degree_5, shift_5] * 100.0
            ),
            "10deg_2cm": float(
                metrics.pose_aps[class_id, degree_10, shift_2] * 100.0
            ),
            "10deg_5cm": float(
                metrics.pose_aps[class_id, degree_10, shift_5] * 100.0
            ),
            "rotation_only_5deg": float(
                metrics.pose_aps[class_id, degree_5, translation_free] * 100.0
            ),
            "translation_only_2cm": float(
                metrics.pose_aps[class_id, rotation_free, shift_2] * 100.0
            ),
        }
    return output


# Per-category REAL275 reference values retained with the experiments.  They are
# evaluation targets only and are never exposed to a network or prediction.
PER_CATEGORY_REFERENCE = {
    "camera": {"5deg_2cm": 2.9, "5deg_5cm": 3.2},
    "laptop": {"5deg_2cm": 63.4, "5deg_5cm": 91.4},
    "bottle": {"5deg_2cm": 52.6, "5deg_5cm": 60.9},
    "bowl": {"5deg_2cm": 85.4, "5deg_5cm": 92.6},
    "can": {"5deg_2cm": 72.5, "5deg_5cm": 80.4},
    "mug": {"5deg_2cm": 35.7, "5deg_5cm": 36.4},
}


def compare_per_category_reference(
    diagnostics: dict[str, dict[str, float]],
) -> dict[str, object]:
    """Compare every REAL275 class against the archived reference table."""

    comparison: dict[str, dict[str, object]] = {}
    for name, reference in PER_CATEGORY_REFERENCE.items():
        if name not in diagnostics:
            raise KeyError(f"REAL275 diagnostics are missing class {name!r}")
        actual = diagnostics[name]
        row: dict[str, object] = {}
        for metric, target in reference.items():
            value = float(actual[metric])
            margin = value - float(target)
            row[metric] = {
                "actual": value,
                "target": float(target),
                "margin": margin,
                "passed": margin > 0.0,
            }
        row["passed_both"] = all(
            bool(row[metric]["passed"]) for metric in reference
        )
        comparison[name] = row
    passed_classes = sum(bool(row["passed_both"]) for row in comparison.values())
    return {
        "source": "archived_REAL275_per_category_reference",
        "strictly_above_required": True,
        "passed_classes": passed_classes,
        "total_classes": len(comparison),
        "passed_all_classes": passed_classes == len(comparison),
        "classes": comparison,
    }
