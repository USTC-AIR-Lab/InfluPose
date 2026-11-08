import math
import tempfile
from argparse import Namespace
from pathlib import Path

import torch

from stage2.train import (
    arguments,
    make_stage2_scheduler,
    require_same_training_dataset,
)


def test_stage2_cli_defaults_use_continuous_full50(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["stage2.train", "--data_path", "/tmp/data", "--output_dir", "/tmp/run"],
    )
    cfg = arguments()
    assert cfg.base_epochs == 30
    assert cfg.epochs == 50
    assert cfg.batch_size == 6
    assert cfg.observability_teacher_chunk == 24
    assert cfg.camera_real_ratio == 1.0
    assert cfg.learning_rate == 3e-4
    assert cfg.head_learning_rate == 7e-4
    assert cfg.ema_decay == 0.999
    assert cfg.irls_iterations == 3
    assert cfg.irls_damping == 1e-3
    assert cfg.max_rotation_step_degrees == 8.0
    assert cfg.max_translation_step == 0.015
    assert cfg.max_log_size_step == 0.12
    assert not hasattr(cfg, "initial_checkpoint")
    assert not any(name.startswith(("real275_", "metric_", "selection_", "continuation_")) for name in vars(cfg))
    assert not hasattr(cfg, "finetune_learning_rate")


def test_stage2_base_and_refinement_schedules_keep_the_original_rates():
    cfg = Namespace(epochs=50, base_epochs=30, learning_rate=3e-4, refine_eta_min=3e-7)
    for refining, epochs, rates, floor in ((False, 30, (3e-4, 7e-4), 3e-5),
                                          (True, 20, (1.5e-5, 3e-5), 3e-7)):
        parameters = [torch.nn.Parameter(torch.tensor(1.)) for _ in range(2)]
        optimizer = torch.optim.AdamW([{"params": [p], "lr": rate} for p, rate in zip(parameters, rates)])
        scheduler = make_stage2_scheduler(optimizer, cfg, refining=refining)
        for _ in range(epochs):
            optimizer.step()
            scheduler.step()
        assert scheduler.last_epoch == epochs
        assert scheduler.T_max == epochs
        assert all(math.isclose(group["lr"], floor) for group in optimizer.param_groups)


def test_checkpoint_lineage_is_bound_to_the_training_dataset():
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        first = root / "dataset-a"
        second = root / "dataset-b"
        first.mkdir()
        second.mkdir()
        cfg = Namespace(data_path=str(first), train_source="CAMERA+Real")
        require_same_training_dataset(
            {"data_path": str(first), "train_source": "CAMERA+Real"},
            cfg,
            context="test checkpoint",
        )
        try:
            require_same_training_dataset(
                {"data_path": str(second), "train_source": "CAMERA+Real"},
                cfg,
                context="test checkpoint",
            )
        except ValueError as error:
            assert "another training dataset" in str(error)
        else:
            raise AssertionError("checkpoint from another dataset was accepted")
