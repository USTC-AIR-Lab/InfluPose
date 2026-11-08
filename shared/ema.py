"""Exponential moving average for model parameters."""

from __future__ import annotations

import torch


class ExponentialMovingAverage:
    def __init__(self, parameters, decay: float, use_num_updates: bool = True):
        if not 0.0 <= decay <= 1.0:
            raise ValueError("decay must lie in [0,1]")
        self.decay = float(decay)
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = [
            parameter.clone().detach()
            for parameter in parameters
            if parameter.requires_grad
        ]
        self.collected_params = []

    def update(self, parameters) -> None:
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        parameters = [parameter for parameter in parameters if parameter.requires_grad]
        with torch.no_grad():
            for shadow, parameter in zip(self.shadow_params, parameters):
                shadow.sub_((1.0 - decay) * (shadow - parameter))

    def copy_to(self, parameters) -> None:
        parameters = [parameter for parameter in parameters if parameter.requires_grad]
        for shadow, parameter in zip(self.shadow_params, parameters):
            parameter.data.copy_(shadow.data)

    def store(self, parameters) -> None:
        self.collected_params = [parameter.clone() for parameter in parameters]

    def restore(self, parameters) -> None:
        for collected, parameter in zip(self.collected_params, parameters):
            parameter.data.copy_(collected.data)

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow_params": self.shadow_params,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.decay = state_dict["decay"]
        self.num_updates = state_dict["num_updates"]
        self.shadow_params = state_dict["shadow_params"]
