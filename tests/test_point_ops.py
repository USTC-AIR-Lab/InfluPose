import torch
import pytest

from shared import point_ops


def test_fps_backend_is_explicit_and_routes_legacy_replay(monkeypatch):
    points = torch.zeros(1, 4, 3)
    monkeypatch.setenv("INFLUPOSE_FPS_BACKEND", "legacy_cuda")
    monkeypatch.setattr(
        point_ops,
        "_legacy_farthest_point_sample",
        lambda values, count: torch.tensor([[0, 3]], dtype=torch.long),
    )
    assert point_ops.farthest_point_sample(points, 2).tolist() == [[0, 3]]

    monkeypatch.setenv("INFLUPOSE_FPS_BACKEND", "unknown")
    with pytest.raises(ValueError, match="INFLUPOSE_FPS_BACKEND"):
        point_ops.farthest_point_sample(points, 2)


def test_native_point_ops_follow_input_order_and_preserve_gradients():
    points = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [1.0, 0.0, 0.0]]]
    )
    fps = point_ops.farthest_point_sample(points, 3)
    assert fps.tolist() == [[0, 3, 2]]

    centers = points[:, :2]
    neighborhoods = point_ops.ball_query(0.25, 4, points, centers)
    assert neighborhoods.tolist() == [[[0, 1, 2, 0], [0, 1, 2, 0]]]

    features = torch.arange(8.0).reshape(1, 2, 4).requires_grad_()
    gathered = point_ops.gather_operation(features, fps)
    grouped = point_ops.grouping_operation(features, neighborhoods)
    (gathered.sum() + grouped.sum()).backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
