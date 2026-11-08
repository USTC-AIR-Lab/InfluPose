from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from stage1.train import atomic_torch_save, capture_rng_state, restore_rng_state


class _RandomDataset(Dataset):
    def __len__(self):
        return 24

    def __getitem__(self, index):
        return (
            index,
            random.random(),
            float(np.random.random()),
            float(torch.rand(())),
        )


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _epoch(loader: DataLoader) -> list[tuple]:
    records = []
    for indices, python_values, numpy_values, torch_values in loader:
        records.extend(
            zip(
                indices.tolist(),
                python_values.tolist(),
                numpy_values.tolist(),
                torch_values.tolist(),
            )
        )
    return records


class Stage1RngCheckpointTests(unittest.TestCase):
    def test_python_numpy_and_torch_cpu_streams_round_trip(self):
        with mock.patch("stage1.train.torch.cuda.is_available", return_value=False):
            _seed(260892)
            state = capture_rng_state()
            expected = (random.random(), np.random.random(), torch.rand(5))

            _seed(7)
            restore_rng_state(state)
            actual = (random.random(), np.random.random(), torch.rand(5))

        self.assertEqual(state["torch_cuda"], [])
        self.assertEqual(expected[0], actual[0])
        self.assertEqual(expected[1], actual[1])
        torch.testing.assert_close(expected[2], actual[2], rtol=0.0, atol=0.0)

    def test_nonpersistent_workers_reconstruct_next_epoch(self):
        with mock.patch("stage1.train.torch.cuda.is_available", return_value=False):
            _seed(260892)
            uninterrupted = DataLoader(
                _RandomDataset(),
                batch_size=5,
                shuffle=True,
                num_workers=2,
                persistent_workers=False,
            )
            _epoch(uninterrupted)
            state = capture_rng_state()
            expected_next_epoch = _epoch(uninterrupted)

            _seed(11)
            resumed = DataLoader(
                _RandomDataset(),
                batch_size=5,
                shuffle=True,
                num_workers=2,
                persistent_workers=False,
            )
            random.random()
            np.random.random()
            torch.rand(17)
            restore_rng_state(state)
            resumed_next_epoch = _epoch(resumed)

        self.assertEqual(expected_next_epoch, resumed_next_epoch)

    def test_missing_rng_field_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "missing fields"):
            restore_rng_state(
                {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch_cpu": torch.get_rng_state(),
                }
            )

    def test_atomic_checkpoint_publish_preserves_previous_target_on_failure(self):
        with tempfile.TemporaryDirectory() as raw:
            target = Path(raw) / "epoch_001.pth"
            atomic_torch_save({"value": torch.tensor([1])}, str(target))
            original_bytes = target.read_bytes()

            def fail_after_partial_write(_payload, temporary):
                with open(temporary, "wb") as stream:
                    stream.write(b"partial")
                raise RuntimeError("injected save failure")

            with mock.patch(
                "stage1.train.torch.save",
                side_effect=fail_after_partial_write,
            ), self.assertRaisesRegex(RuntimeError, "injected"):
                atomic_torch_save({"value": torch.tensor([2])}, str(target))

            self.assertEqual(target.read_bytes(), original_bytes)
            self.assertEqual(list(target.parent.glob("epoch_001.pth.tmp.*")), [])


if __name__ == "__main__":
    unittest.main()
