"""Training-only restriction for issue175; evaluation uses the unchanged DQN.

Arm provenance lives in the immutable campaign binding and each training row.
Checkpoint serialization is unchanged. Every reload must reinstall this guard;
loading a treatment checkpoint for unrestricted training is a new intervention.
"""

from __future__ import annotations

import torch

FIRST = "layers.0.weight"
PREFIX = 26


def validate_frozen(state, anchor):
    if state.keys() != anchor.keys() or state[FIRST].shape != (64, 39):
        raise ValueError("Expected archived 39-input network")
    for name, value in state.items():
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite online parameter")
        actual = value[:, :PREFIX] if name == FIRST else value
        expected = anchor[name][:, :PREFIX] if name == FIRST else anchor[name]
        if not torch.equal(actual, expected):
            raise ValueError("Frozen online parameter moved: " + name)


class FrozenOpponentInputs:
    """Mask before clipping and fail closed on frozen Adam state or weight drift."""

    def __init__(self, learner, anchor):
        self.network = learner.online_network
        self.optimizer = learner.optimizer
        self.anchor = {k: v.detach().clone() for k, v in anchor.items()}
        self.parameters = dict(self.network.named_parameters())
        if not isinstance(self.optimizer, torch.optim.Adam):
            raise ValueError("Expected registered Adam optimizer")
        if any(g["weight_decay"] != 0 for g in self.optimizer.param_groups):
            raise ValueError("Weight decay would move frozen entries")
        self.validate()
        for name, parameter in self.parameters.items():
            parameter.requires_grad_(name == FIRST)
            parameter.grad = None
        self.gradient_handle = self.parameters[FIRST].register_hook(self.mask_gradient)
        self.before_handle = self.optimizer.register_step_pre_hook(self.before_step)
        self.after_handle = self.optimizer.register_step_post_hook(self.after_step)

    @staticmethod
    def mask_gradient(gradient):
        if not torch.isfinite(gradient).all():
            raise ValueError("Nonfinite gradient")
        result = gradient.clone()
        result[:, :PREFIX] = 0
        return result

    def validate(self):
        validate_frozen(self.network.state_dict(), self.anchor)
        for name, parameter in self.parameters.items():
            for key, value in self.optimizer.state.get(parameter, {}).items():
                if not isinstance(value, torch.Tensor):
                    continue
                if not torch.isfinite(value).all():
                    raise ValueError("Nonfinite optimizer state")
                if key == "step":
                    continue
                if value.shape != parameter.shape:
                    raise ValueError("Unexpected optimizer moment shape")
                frozen = value[:, :PREFIX] if name == FIRST else value
                if torch.count_nonzero(frozen):
                    raise ValueError("Nonzero frozen Adam moment")

    def before_step(self, optimizer, args, kwargs):
        self.validate()
        for name, parameter in self.parameters.items():
            if parameter.grad is None:
                continue
            if name != FIRST or torch.count_nonzero(parameter.grad[:, :PREFIX]):
                raise ValueError("Frozen parameter received gradient")

    def after_step(self, optimizer, args, kwargs):
        self.validate()
