from types import SimpleNamespace

import pytest
import torch

from stage2.auxiliary_losses import (
    AUXILIARY_DEFAULTS, detached_state_dict, frozen_reference,
    reference_enabled, reference_non_regression_loss, robust_partial_view,
    validate_auxiliary_config, view_consistency_loss,
)


def prediction(translation, size=0.2, requires_grad=False):
    pose = torch.tensor([[1., 0., 0., 0., 1., 0., translation, 0., 0.8]], requires_grad=requires_grad)
    return {"pose": pose, "size": torch.full((1, 3), size, requires_grad=requires_grad)}


def test_view_target_is_detached_but_pose_and_size_learn():
    primary, secondary = prediction(0., requires_grad=True), prediction(0.03, 0.25, True)
    loss = view_consistency_loss(secondary, primary, torch.tensor([2]), torch.tensor([1]), 0.75)
    loss.backward()
    assert loss.item() > 0
    assert secondary["pose"].grad[:, 6:].abs().sum() > 0
    assert secondary["size"].grad.abs().sum() > 0
    assert primary["pose"].grad is None and primary["size"].grad is None


def test_non_regression_rewards_improvement_and_enforces_tolerances():
    target, reference = prediction(0.), prediction(0.02, requires_grad=True)
    def regrets(student):
        return reference_non_regression_loss(
            student, reference, target["pose"], target["size"],
            torch.tensor([2]), torch.tensor([1]),
            rotation_margin_degrees=0.25, translation_margin_cm=0.10,
            iou_margin=0.01,
        )
    better_pose, better_iou, _ = regrets(prediction(0.01))
    assert better_pose.item() == 0 and better_iou.item() == 0
    within_pose, within_iou, _ = regrets(prediction(0.0205))
    assert within_pose.item() == 0 and within_iou.item() == 0
    student = prediction(0.04, requires_grad=True)
    pose_loss, iou_loss, stats = regrets(student)
    assert pose_loss.item() == pytest.approx(2 * (0.04 - 0.02 - 0.001))
    assert iou_loss.item() > 0
    assert stats["reference_translation_regression_cm"].item() == pytest.approx(1.9)
    (pose_loss + iou_loss).backward()
    assert torch.isfinite(student["pose"].grad).all()
    assert student["pose"].grad[0, 6] > 0
    assert reference["pose"].grad is None and reference["size"].grad is None


def test_frozen_reference_and_cpu_snapshot_do_not_alias_student():
    model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Dropout(0.5))
    reference = frozen_reference(model)
    snapshot = detached_state_dict(reference)
    assert not reference.training and all(not p.requires_grad for p in reference.parameters())
    with torch.no_grad():
        next(model.parameters()).add_(3)
    assert not torch.equal(next(model.parameters()), next(reference.parameters()))
    restored = frozen_reference(model, snapshot)
    for key, value in snapshot.items():
        assert torch.equal(value, restored.state_dict()[key])
    with torch.no_grad():
        next(reference.parameters()).add_(1)
    assert not torch.equal(next(reference.parameters()), snapshot['0.weight'])


def test_auxiliary_loss_defaults_are_enabled_for_refinement():
    cfg = SimpleNamespace(**AUXILIARY_DEFAULTS)
    validate_auxiliary_config(cfg)
    assert reference_enabled(cfg) and cfg.view_consistency_weight > 0
    cfg.view_fraction = 0
    with pytest.raises(ValueError, match="positive view_fraction"):
        validate_auxiliary_config(cfg)


def test_second_view_keeps_cardinality_and_no_corruption_is_identity():
    points = torch.randn(3, 32, 3)
    assert torch.equal(points, robust_partial_view(points, dropout_max=0, noise_std=0, axial_noise_std=0))
    second = robust_partial_view(points, dropout_max=.25, noise_std=.001, axial_noise_std=.002)
    assert second.shape == points.shape and torch.isfinite(second).all()
    assert not torch.equal(points, second)
