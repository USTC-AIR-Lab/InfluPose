"""Export inference parameters and portable metadata from a training checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_metadata(value):
    if isinstance(value, dict):
        return {
            key: portable_metadata(item)
            for key, item in value.items()
            if key not in {"hostname", "username", "author", "email", "command", "argv"}
        }
    if isinstance(value, (tuple, list)):
        return [portable_metadata(item) for item in value]
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        if value.startswith("/") or re.match(r"^[A-Za-z]:[\\/]", value):
            return "artifacts/" + Path(value.replace("\\", "/")).name
        if re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", value):
            return "anonymous"
    return value


def export(source: Path, output: Path) -> dict:
    state = torch.load(source, map_location="cpu", weights_only=False)
    selected = state.get("ema_model_state_dict", state["model_state_dict"])
    selected = {key: value.detach().cpu().clone() for key, value in selected.items()}
    retained = (
        "format", "epoch", "epoch_index_base", "global_step", "architecture", "class_prior_audit",
        "contract_audit", "latent_contract", "dataset_contract", "provenance",
    )
    result = {key: portable_metadata(state[key]) for key in retained if key in state}
    result["model_state_dict"] = selected
    result["ema_model_state_dict"] = selected
    result["recommended_evaluation_weights"] = "ema"
    result["release_export"] = {
        "kind": "inference_parameters",
        "resumable": False,
        "selected_weights": "ema",
        "tensor_values_preserved": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)
    restored = torch.load(output, map_location="cpu", weights_only=False)
    assert selected.keys() == restored["ema_model_state_dict"].keys()
    for key in selected:
        if not torch.equal(selected[key], restored["ema_model_state_dict"][key]):
            raise RuntimeError("Export changed tensor " + key)
    return {
        "file": output.name,
        "sha256": sha256(output),
        "bytes": output.stat().st_size,
        "weights": "ema",
        "resumable": False,
        "tensor_count": len(selected),
        "tensor_values_verified": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    cfg = parser.parse_args()
    if cfg.source.resolve() == cfg.output.resolve() or cfg.output.exists():
        parser.error("--output must be a new file")
    print(json.dumps(export(cfg.source, cfg.output), indent=2))


if __name__ == "__main__":
    main()
