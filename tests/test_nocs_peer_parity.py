"""Offline regression against pinned official GPV-Pose evaluator outputs."""

import json
from pathlib import Path

import numpy as np

from shared.real275_metrics import compute_real275_3d_map


def test_box_and_pose_ap_match_the_official_peer_fixture(tmp_path):
    fixture = json.loads((Path(__file__).parent / "fixtures/nocs_peer_parity.json").read_text())
    frames = [{key: np.asarray(value) for key, value in frame.items()} for frame in fixture["frames"]]
    result = compute_real275_3d_map(
        frames, str(tmp_path),
        iou_thresholds=tuple(fixture["iou_thresholds"]),
        degree_thresholds=tuple(fixture["degree_thresholds"]),
        shift_thresholds=tuple(fixture["shift_thresholds"]),
    )
    np.testing.assert_allclose(result.iou_aps, fixture["expected_iou_aps"], atol=1e-7, rtol=0)
    np.testing.assert_allclose(result.pose_aps, fixture["expected_pose_aps"], atol=1e-7, rtol=0)
