"""Minimal NOCS 3D-IoU and pose-matching primitives used by REAL275."""

from __future__ import annotations

import math

import numpy as np


def get_3d_bbox(size, shift=0):
    bbox = np.array(
        [
            [+size[0] / 2, +size[1] / 2, +size[2] / 2],
            [+size[0] / 2, +size[1] / 2, -size[2] / 2],
            [-size[0] / 2, +size[1] / 2, +size[2] / 2],
            [-size[0] / 2, +size[1] / 2, -size[2] / 2],
            [+size[0] / 2, -size[1] / 2, +size[2] / 2],
            [+size[0] / 2, -size[1] / 2, -size[2] / 2],
            [-size[0] / 2, -size[1] / 2, +size[2] / 2],
            [-size[0] / 2, -size[1] / 2, -size[2] / 2],
        ]
    ) + shift
    return bbox.transpose()


def transform_coordinates_3d(coordinates, transform):
    if coordinates.shape[0] != 3:
        raise ValueError("coordinates must have shape [3,N]")
    homogeneous = np.vstack(
        (coordinates, np.ones((1, coordinates.shape[1]), dtype=np.float32))
    )
    transformed = transform @ homogeneous
    return transformed[:3] / transformed[3:4]


def compute_3d_iou(
    first_transform,
    second_transform,
    first_size,
    second_size,
    first_class,
    second_class,
    handle_visibility,
):
    """Match the archived NOCS implementation, including 20-way y symmetry."""

    def asymmetric(first, second):
        first_box = transform_coordinates_3d(get_3d_bbox(first_size, 0), first)
        second_box = transform_coordinates_3d(get_3d_bbox(second_size, 0), second)
        # Keep the archived protocol byte-for-byte: it reduces axis 0 here.
        first_max, first_min = np.amax(first_box, axis=0), np.amin(first_box, axis=0)
        second_max, second_min = np.amax(second_box, axis=0), np.amin(second_box, axis=0)
        overlap_min = np.maximum(first_min, second_min)
        overlap_max = np.minimum(first_max, second_max)
        intersection = (
            0.0
            if np.amin(overlap_max - overlap_min) < 0
            else float(np.prod(overlap_max - overlap_min))
        )
        union = (
            np.prod(first_max - first_min)
            + np.prod(second_max - second_min)
            - intersection
        )
        return float(intersection / union)

    if first_transform is None or second_transform is None:
        return -1.0
    symmetric = (
        first_class in ("bottle", "bowl", "can") and first_class == second_class
    ) or (
        first_class == "mug"
        and first_class == second_class
        and handle_visibility == 0
    )
    if not symmetric:
        return asymmetric(first_transform, second_transform)
    best = 0.0
    for index in range(20):
        theta = 2.0 * math.pi * index / 20.0
        yaw = np.array(
            [
                [math.cos(theta), 0, math.sin(theta), 0],
                [0, 1, 0, 0],
                [-math.sin(theta), 0, math.cos(theta), 0],
                [0, 0, 0, 1],
            ]
        )
        best = max(best, asymmetric(first_transform @ yaw, second_transform))
    return best


def compute_IoU_matches(
    gt_class_ids,
    gt_transforms,
    gt_sizes,
    gt_handle_visibility,
    pred_class_ids,
    pred_transforms,
    pred_sizes,
    pred_scores,
    synset_names,
    iou_thresholds,
    score_threshold=0,
):
    prediction_count, target_count = len(pred_class_ids), len(gt_class_ids)
    indices = np.zeros(0)
    if prediction_count:
        indices = np.argsort(pred_scores)[::-1]
        pred_class_ids = pred_class_ids[indices].copy()
        pred_sizes = pred_sizes[indices].copy()
        pred_transforms = pred_transforms[indices].copy()
    overlaps = np.zeros((prediction_count, target_count), dtype=np.float32)
    for prediction in range(prediction_count):
        for target in range(target_count):
            overlaps[prediction, target] = compute_3d_iou(
                pred_transforms[prediction],
                gt_transforms[target],
                pred_sizes[prediction],
                gt_sizes[target],
                synset_names[pred_class_ids[prediction]],
                synset_names[gt_class_ids[target]],
                gt_handle_visibility[target],
            )
    pred_matches = -np.ones((len(iou_thresholds), prediction_count))
    gt_matches = -np.ones((len(iou_thresholds), target_count))
    for threshold_index, threshold in enumerate(iou_thresholds):
        for prediction in range(indices.shape[0]):
            ordered = np.argsort(overlaps[prediction])[::-1]
            low = np.where(overlaps[prediction, ordered] < score_threshold)[0]
            if low.size:
                ordered = ordered[: low[0]]
            for target in ordered:
                if gt_matches[threshold_index, target] > -1:
                    continue
                iou = overlaps[prediction, target]
                if iou < threshold:
                    break
                if pred_class_ids[prediction] != gt_class_ids[target]:
                    continue
                if iou > threshold:
                    gt_matches[threshold_index, target] = prediction
                    pred_matches[threshold_index, prediction] = target
                    break
    return gt_matches, pred_matches, overlaps, indices


def compute_rt_errors(
    first_transform,
    second_transform,
    class_id,
    handle_visibility,
    synset_names,
):
    if first_transform is None or second_transform is None:
        return np.array([-1.0, -1.0])
    first_rotation = first_transform[:3, :3] / np.cbrt(
        np.linalg.det(first_transform[:3, :3])
    )
    second_rotation = second_transform[:3, :3] / np.cbrt(
        np.linalg.det(second_transform[:3, :3])
    )
    class_name = synset_names[class_id]
    if class_name in ("bottle", "can", "bowl") or (
        class_name == "mug" and handle_visibility == 0
    ):
        y_axis = np.array([0, 1, 0])
        first_y, second_y = first_rotation @ y_axis, second_rotation @ y_axis
        cosine = first_y.dot(second_y) / (
            np.linalg.norm(first_y) * np.linalg.norm(second_y)
        )
    else:
        relative = first_rotation @ second_rotation.transpose()
        cosine = (np.trace(relative) - 1.0) / 2.0
    angle = np.arccos(np.clip(cosine, -1.0, 1.0)) * 180.0 / np.pi
    shift = np.linalg.norm(first_transform[:3, 3] - second_transform[:3, 3]) * 100.0
    return np.array((angle, shift))


def compute_RT_overlaps(
    gt_class_ids,
    gt_transforms,
    gt_handle_visibility,
    pred_class_ids,
    pred_transforms,
    synset_names,
):
    overlaps = np.zeros((len(pred_class_ids), len(gt_class_ids), 2))
    for prediction in range(len(pred_class_ids)):
        for target in range(len(gt_class_ids)):
            overlaps[prediction, target] = compute_rt_errors(
                pred_transforms[prediction],
                gt_transforms[target],
                gt_class_ids[target],
                gt_handle_visibility[target],
                synset_names,
            )
    return overlaps


def compute_RT_matches(
    overlaps, pred_class_ids, gt_class_ids, degree_thresholds, shift_thresholds
):
    pred_matches = -np.ones(
        (len(degree_thresholds), len(shift_thresholds), len(pred_class_ids))
    )
    gt_matches = -np.ones(
        (len(degree_thresholds), len(shift_thresholds), len(gt_class_ids))
    )
    if not len(pred_class_ids) or not len(gt_class_ids):
        return gt_matches, pred_matches
    for degree_index, degree in enumerate(degree_thresholds):
        for shift_index, shift in enumerate(shift_thresholds):
            for prediction in range(len(pred_class_ids)):
                ordered = np.argsort(np.sum(overlaps[prediction], axis=-1))
                for target in ordered:
                    if (
                        gt_matches[degree_index, shift_index, target] > -1
                        or pred_class_ids[prediction] != gt_class_ids[target]
                        or overlaps[prediction, target, 0] > degree
                        or overlaps[prediction, target, 1] > shift
                    ):
                        continue
                    gt_matches[degree_index, shift_index, target] = prediction
                    pred_matches[degree_index, shift_index, prediction] = target
                    break
    return gt_matches, pred_matches


def compute_ap_and_acc(pred_matches, pred_scores, gt_matches):
    score_indices = np.argsort(pred_scores)[::-1]
    pred_matches = pred_matches[score_indices]
    precision = np.cumsum(pred_matches > -1) / (np.arange(len(pred_matches)) + 1)
    recall = np.cumsum(pred_matches > -1).astype(np.float32) / len(gt_matches)
    precision = np.concatenate(([0], precision, [0]))
    recall = np.concatenate(([0], recall, [1]))
    for index in range(len(precision) - 2, -1, -1):
        precision[index] = np.maximum(precision[index], precision[index + 1])
    changes = np.where(recall[:-1] != recall[1:])[0] + 1
    ap = np.sum((recall[changes] - recall[changes - 1]) * precision[changes])
    accuracy = np.sum(pred_matches > -1) / len(pred_matches)
    return ap, accuracy
