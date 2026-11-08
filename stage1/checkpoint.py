"""Load the frozen complete-cloud teacher independently of evaluation datasets."""

from __future__ import annotations

import torch

from stage1.model import Stage1CompleteCloudRAE

CHECKPOINT_FORMAT = "STAGE1_COMPLETE_CLOUD_RAE_V1"


def load_model(
    path: str, device: str
) -> tuple[Stage1CompleteCloudRAE, dict]:
    state = torch.load(path, map_location=device, weights_only=False)
    if state.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("checkpoint is not the archived Stage1 architecture")
    provenance = state.get("provenance", {})
    if provenance.get("trained_from_scratch") is not True:
        raise ValueError("Stage1 checkpoint does not document scratch training")
    if provenance.get("external_pretrained_weights", True):
        raise ValueError("Stage1 checkpoint violates its no-pretrained-weight contract")
    if state.get("class_prior_audit", {}).get("passed") is not True:
        raise ValueError("Stage1 checkpoint lacks a passing category-prior audit")
    model = Stage1CompleteCloudRAE(**state["architecture"]).to(device)
    model.load_state_dict(
        state.get("ema_model_state_dict", state["model_state_dict"]), strict=True
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, state
