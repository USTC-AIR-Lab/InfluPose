import json
import os
from pathlib import Path
import shlex
import subprocess

import pytest
import torch

from stage2 import train


@pytest.mark.parametrize("option", [
    "--real275_eval_every", "--metric_finetune_checkpoint",
    "--continuation_checkpoint", "--initial_checkpoint",
    "--strict_pose_weight", "--strict_rotation_temperature_degrees",
    "--strict_translation_temperature_cm",
    "--refine_pose_5deg_2cm_weight", "--refine_pose_5deg_5cm_weight",
    "--refine_pose_10deg_2cm_weight", "--refine_pose_10deg_5cm_weight",
    "--refine_rotation_focus_weight", "--refine_asymmetric_sample_weight",
    "--refine_iou50_weight", "--refine_iou75_weight",
    "--refine_iou_temperature", "--refine_iou_threshold_weight",
])
def test_removed_training_modes_are_rejected(monkeypatch, option):
    monkeypatch.setattr("sys.argv", ["train", "--data_path", "data", "--output_dir", "run", option, "1"])
    with pytest.raises(SystemExit) as error:
        train.arguments()
    assert error.value.code == 2


def test_launcher_uses_one_fifty_epoch_run_and_resumes_it(tmp_path):
    root = Path(__file__).resolve().parents[1]
    data = tmp_path / "data"
    for name in ("CAMERA", "Real", "obj_models"):
        (data / name).mkdir(parents=True)
    teacher = tmp_path / "teacher.pth"
    teacher.write_bytes(b"dry-run fixture")
    audit = tmp_path / "audit.json"
    audit.write_text('{"passed":true}')
    ops = tmp_path / "point_ops"
    ops.mkdir()
    (ops / "pointnet2_utils.py").touch()
    (ops / "pointnet2_cuda.fixture.so").touch()
    binaries = tmp_path / "bin"
    binaries.mkdir()
    gpu = binaries / "nvidia-smi"
    gpu.write_text("#!/bin/sh\nexit 0\n")
    gpu.chmod(0o755)
    output = tmp_path / "training"
    env = dict(os.environ, PATH=str(binaries) + os.pathsep + os.environ["PATH"],
               LEGACY_POINT_OPS_PATH=str(ops), GPU_COUNT="6", BATCH_SIZE="6")
    command = ["bash", str(root / "scripts/train_stage2.sh"), "--dry-run", "--data-path", str(data),
               "--stage1-checkpoint", str(teacher), "--stage1-audit", str(audit),
               "--output-dir", str(output), "--gpu-ids", "0,1,2,3,4,5"]
    def launches():
        result = subprocess.run(command, env=env, text=True, capture_output=True, check=True)
        return [shlex.split(line[len("DRY-RUN:"):]) for line in result.stdout.splitlines() if line.startswith("DRY-RUN:")]
    commands = launches()
    assert len(commands) == 1
    for cmd in commands:
        assert cmd[cmd.index("--view_consistency_weight") + 1] == "0.5"
        assert cmd[cmd.index("--reference_pose_weight") + 1] == "2.0"
        assert cmd[cmd.index("--reference_iou_weight") + 1] == "2.0"
    assert commands[0][commands[0].index("--epochs") + 1] == "50"
    assert commands[0][commands[0].index("--base_epochs") + 1] == "30"
    assert "--initial_checkpoint" not in commands[0]
    assert all(not any(arg.startswith(("--metric_", "--real275_", "--selection_")) for arg in cmd) for cmd in commands)
    assert not output.exists()
    (output / "full50").mkdir(parents=True)
    (output / "full50" / "latest.pth").touch()
    commands = launches()
    assert len(commands) == 1
    assert commands[0][commands[0].index("--resume") + 1] == str(output / "full50" / "latest.pth")
    assert "--initial_checkpoint" not in commands[0]


@pytest.mark.parametrize("interrupt_after", [2, 3])
def test_training_transition_and_resume_preserve_epoch30_reference(tmp_path, monkeypatch, interrupt_after):
    monkeypatch.setattr("sys.argv", ["train", "--data_path", str(tmp_path / "data"), "--output_dir", str(tmp_path / "run")])
    cfg = train.arguments()
    cfg.device, cfg.amp, cfg.fps_backend = "cpu", "none", "torch"
    cfg.batch_size, cfg.num_workers = 36, 0
    cfg.token_width, cfg.anchor_count, cfg.relation_layers, cfg.heads = 24, 8, 1, 4
    cfg.dropout, cfg.normals_k = 0.0, 4
    cfg.epochs, cfg.base_epochs, cfg.max_train_batches = 4, 2, 1
    cfg.allow_noncanonical_schedule = True
    cfg.observability_fraction = 0.0
    teacher_path = tmp_path / "teacher.pth"
    teacher_path.write_bytes(b"synthetic teacher")
    cfg.stage1_checkpoint = str(teacher_path)
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps({"passed": True, "checkpoint_sha256": train.file_sha256(str(teacher_path))}))
    cfg.observability_audit = str(audit_path)
    generator = torch.Generator().manual_seed(8)
    points = torch.randn(2, 32, 3, generator=generator) * .04 + torch.tensor([0., 0., .8])
    pose = torch.tensor([[1., 0., 0., 0., 1., 0., 0., 0., .8]]).repeat(2, 1)
    batch = {"pts": points, "pts_center": points.mean(1), "gt_pose": pose,
             "gt_size": torch.full((2, 3), .15), "id": torch.tensor([2, 4]),
             "handle_visibility": torch.ones(2, dtype=torch.long)}
    class InterruptedLoader:
        calls = 0
        def __iter__(self):
            self.calls += 1
            if self.calls == interrupt_after + 1:
                raise RuntimeError("simulated interruption")
            return iter([batch])
        def __len__(self):
            return 1
    loader = InterruptedLoader()
    ratios = []
    def data_loader(**kwargs):
        ratios.append(kwargs["camera_real_ratio"])
        return loader
    monkeypatch.setattr(train, "arguments", lambda: cfg)
    monkeypatch.setattr(train, "load_stage1", lambda *args: (torch.nn.Identity(), {"epoch": 0}))
    monkeypatch.setattr(train, "get_data_loaders", data_loader)
    monkeypatch.setattr(train, "process_batch", lambda raw, **kwargs: raw)
    torch.set_num_threads(1)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        train.main()
    output = Path(cfg.output_dir)
    base = torch.load(output / "epoch_002.pth", weights_only=False)
    assert base["reference_model_state_dict"] is None
    assert base["training_phase"] == "base"
    assert base["scheduler_state_dict"]["T_max"] == 2
    latest = torch.load(output / "latest.pth", weights_only=False)
    assert latest["global_step"] == interrupt_after
    assert ratios == [1., .75]
    cfg.resume = str(output / "latest.pth")
    monkeypatch.setattr(train, "get_data_loaders", lambda **kwargs: [batch])
    if interrupt_after > cfg.base_epochs:
        def reject_transition(*args):
            raise AssertionError("resuming refinement must not reset its optimizer or reference")
        monkeypatch.setattr(train, "begin_refinement", reject_transition)
    train.main()
    final = torch.load(output / "final.pth", weights_only=False)
    reference = final["reference_model_state_dict"]
    assert final["training_phase"] == "refinement"
    assert final["reference_epoch"] == 2
    assert all(torch.equal(value, reference[key]) for key, value in base["ema_model_state_dict"].items())
    assert any(not torch.equal(value, final["model_state_dict"][key]) for key, value in reference.items())
    assert final["epoch"] == 3 and final["global_step"] == 4
    assert final["ema_state_dict"]["num_updates"] == 2
    assert final["scheduler_state_dict"]["last_epoch"] == 2
    assert final["recommended_evaluation_weights"] == "ema"
    assert not (output / "best.pth").exists()
    history = [json.loads(line) for line in (output / "history.jsonl").read_text().splitlines()]
    assert len(history) == 4
    assert all(row["view_examples"] == 0 and row["reference_pose"] == 0 for row in history[:2])
    assert all(row["view_examples"] > 0 and row["view_consistency"] > 0 and row["robust_view_pose"] > 0 for row in history[2:])
    assert all("reference_rotation_regression_deg" in row for row in history[2:])
    final["training_protocol"] = "RANDOM_INIT_BASE30_REFINE20_V5"
    torch.save(final, output / "latest.pth")
    with pytest.raises(ValueError, match="different training protocol"):
        train.main()
