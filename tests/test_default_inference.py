import json
import pickle

import numpy as np
import pytest
import torch

from scripts import infer
from stage2 import evaluate_real275 as evaluate


class Predictor:
    irls_iterations = 3

    def __call__(self, points, *args, **kwargs):
        assert self.irls_iterations == 1
        count = len(points)
        pose = torch.zeros((count, 9), dtype=torch.float32)
        pose[:, 0] = 1
        pose[:, 4] = 1
        pose[:, 8] = 1
        output = {key: torch.ones((count, 1)) for key in (
            'inlier_logits', 'axis_precision', 'information', 'solver_weights',
            'solver_gate', 'solver_condition', 'solver_residual_rms')}
        output.update(pose=pose, size=torch.ones((count, 3)) * .1)
        return output


def features(points, centers, k):
    assert k == 16
    return torch.zeros_like(points), torch.zeros_like(points)


def test_frame_inference_returns_network_pose_and_size(monkeypatch):
    points = np.random.default_rng(1).normal(size=(2, 1024, 3)).astype(np.float32)
    before = points.copy()
    monkeypatch.setattr(infer, 'estimate_camera_observation_features', features)
    rt, size = infer.predict_frame(Predictor(), points, np.array([1, 6]), 'cpu')
    np.testing.assert_array_equal(rt[:, 2, 3], [1., 1.])
    np.testing.assert_array_equal(rt[:, :3, :3], np.repeat(np.eye(3)[None], 2, axis=0))
    np.testing.assert_allclose(size, .1)
    np.testing.assert_array_equal(points, before)


@pytest.mark.parametrize("epoch_metadata, expected_epoch", [
    ({}, 0),
    ({"epoch": 49}, 50),
    ({"epoch": 49, "epoch_index_base": 0}, 50),
    ({"epoch": 50, "epoch_index_base": 1}, 50),
])
def test_evaluator_scores_and_saves_unmodified_network_predictions(
    tmp_path, monkeypatch, epoch_metadata, expected_epoch
):
    monkeypatch.setattr('sys.argv', ['evaluate', '--output_dir', str(tmp_path / 'out')])
    cfg = evaluate.arguments()
    assert cfg.evaluation_irls_iterations == 1
    cfg.device = 'cpu'
    cfg.fps_backend = 'torch'
    cfg.data_path = str(tmp_path / 'no_depth_frames')
    cfg.checkpoint = str(tmp_path / 'weights.pth')
    (tmp_path / 'weights.pth').write_bytes(b'test-fixture')
    cfg.detector_cache = str(tmp_path / 'input.pkl')
    image = 'Real/test/scene_1/0000'
    item = {'result': {'pred_class_ids': np.array([1]), 'pred_scores': np.array([.9]),
                      'pred_bboxes': np.array([[0, 0, 10, 10]])},
            'valid_pts': [np.zeros((1024, 3), dtype=np.float32)], 'cat_id': [0], 'valid_inst': [0]}
    with open(cfg.detector_cache, 'wb') as stream:
        pickle.dump({image: item}, stream)
    monkeypatch.setattr(evaluate, 'arguments', lambda: cfg)
    state = {'resolved_evaluation_weights': 'ema', 'training_config': {'num_points': 1024}}
    state.update(epoch_metadata)
    monkeypatch.setattr(evaluate, 'load_model', lambda *args: (Predictor(), state))
    monkeypatch.setattr(evaluate, 'estimate_camera_observation_features', features)
    calls = []
    def score(results, directory):
        assert results[0]['pred_RTs'][0, 2, 3] == 1.
        np.testing.assert_allclose(results[0]['pred_scales'], .1)
        for key in ('pred_class_ids', 'pred_scores', 'pred_bboxes'):
            np.testing.assert_array_equal(results[0][key], item['result'][key])
        calls.append('score')
        return {}
    monkeypatch.setattr(evaluate, 'compute_real275_3d_map', score)
    monkeypatch.setattr(evaluate, 'summary_from_metrics', lambda _: {'5deg_2cm': 0.})
    monkeypatch.setattr(evaluate, 'per_class_diagnostics', lambda _: {})
    monkeypatch.setattr(evaluate, 'compare_per_category_reference', lambda _: {})
    evaluate.main()
    assert calls == ['score']
    with open(tmp_path / 'out/stage2_predictions.pkl', 'rb') as stream:
        final = pickle.load(stream)
    assert final[image]['result']['pred_RTs'][0, 2, 3] == 1.
    summary = json.loads((tmp_path / 'out/evaluation_summary.json').read_text())
    assert summary['full_evaluation'] is True
    assert summary['evaluation_irls_iterations'] == 1
    assert summary['checkpoint_epoch'] == expected_epoch
    assert summary['input'] == 'detector_partial_absolute_camera_point_cloud_with_normals_and_view_evidence'
