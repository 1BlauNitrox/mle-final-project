"""Neural-network components for DagobertDuckDQN"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import ACTIONS, DEFAULT_CONFIG, DQNConfig
from .replay import ReplayBatch

CPU_DEVICE = torch.device("cpu")

@dataclass(frozen=True)
class TrainingResult:
    """Diagnostics produced by one optimizer update."""

    loss: float
    mean_abs_td_error: float
    target_synchronized: bool

class QNetwork(nn.Module):
    """Small feed-forward network that estimates action values"""

    def __init__(self, config: DQNConfig = DEFAULT_CONFIG) -> None:
        super().__init__()
        self.config = config

        dimensions = (
            config.input_dim,
            *config.hidden_sizes,
            config.output_dim
        )

        layers: list[nn.Module] = []

        for input_size, output_size in zip(
            dimensions[:-1],
            dimensions[1:],
            strict=True
        ): 
            layers.append(nn.Linear(input_size, output_size))

            if output_size != config.output_dim:
                layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Calculate Q-Values for one state or a state batch"""
        if inputs.ndim not in (1, 2):
            raise ValueError("Network input must be a state or state batch")
        
        if inputs.shape[-1] != self.config.input_dim:
            raise ValueError(
                f"Expected {self.config.input_dim} input features,"
                f"got {inputs.shape[-1]}"
            )
        
        if inputs.dtype != torch.float32:
            raise ValueError("Network input must use float32")
        
        if inputs.device.type != "cpu":
            raise ValueError("DagobertDuckDQN supports CPU tensors only")
        
        return self.layers(inputs)
    
def build_q_network(
    config: DQNConfig = DEFAULT_CONFIG,
    *,
    seed: int
) -> QNetwork:
    """Build a deterministically initialized CPU network"""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        network = QNetwork(config)

    return network.to(CPU_DEVICE)

def select_action(
    *,
    network: QNetwork,
    state: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
) -> str:
    """Select an action using seeded epsilon-greedy exploration"""
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0, 1].")

    state_values = np.asarray(state)

    if state_values.shape != (network.config.input_dim,):
        raise ValueError("state has an incompatible shape.")

    if state_values.dtype != np.float32:
        raise ValueError("state must use float32.")
    
    if rng.random() < epsilon:
        action_index = int(rng.integers(len(ACTIONS)))
        return ACTIONS[action_index]
    
    state_tensor = torch.from_numpy(state_values).to(CPU_DEVICE)

    with torch.no_grad():
        q_values = network(state_tensor)

    q_values_array = q_values.detach().cpu().numpy()

    if not np.all(np.isfinite(q_values_array)):
        raise ValueError("Network produced non-finite Q-Values")
    
    maximum = np.max(q_values_array)
    best_indices = np.flatnonzero(q_values_array == maximum)
    selected_index = int(rng.choice(best_indices))

    return ACTIONS[selected_index]

def compute_bellman_targets(
    *,
    rewards: torch.Tensor,
    next_q_values: torch.Tensor,
    terminals: torch.Tensor,
    discount_factor: float,
) -> torch.Tensor:
    """Calculate fixed DQN targets for one transition batch"""
    if not 0.0 <= discount_factor <= 1.0:
        raise ValueError("discount factor must be in [0,1]")
    
    if rewards.ndim != 1:
        raise ValueError("rewards must be one-dimensional")
    
    batch_size = rewards.shape[0]

    if next_q_values.shape != (batch_size, len(ACTIONS)):
        raise ValueError("next_q_values have an incompatible shape")
    
    if terminals.shape != (batch_size, ):
        raise ValueError("terminals have an incompatibel shape")
    
    if rewards.dtype != torch.float32:
        raise ValueError("rewards must use float32")
    
    if next_q_values.dtype != torch.float32:
        raise ValueError("next_q_values must use float32")
    
    if terminals.dtype != torch.bool:
        raise ValueError("terminals must use bool")
    
    with torch.no_grad():
        maximum_next_q_values = next_q_values.max(dim=1).values
        bootstrap_mask = (~terminals).to(dtype=torch.float32)

        return(
            rewards
            + discount_factor
            * bootstrap_mask
            * maximum_next_q_values
        )
    
class DQNLearner:
    """Online network, target network and optimizer for DQN training."""

    def __init__(
        self,
        *,
        config: DQNConfig = DEFAULT_CONFIG,
        seed: int,
    ) -> None:
        self.config = config

        self.online_network = build_q_network(
            config,
            seed=seed,
        )
        self.target_network = deepcopy(self.online_network)
        self.target_network.eval()
        self.target_network.requires_grad_(False)

        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(),
            lr=config.learning_rate,
        )
        self.update_steps = 0

    def train_batch(self, batch: ReplayBatch) -> TrainingResult:
        """Apply one mini-batch DQN update."""
        expected_state_shape = (
            self.config.batch_size,
            self.config.input_dim,
        )

        if batch.states.shape != expected_state_shape:
            raise ValueError(
                f"Expected states with shape {expected_state_shape}, "
                f"got {batch.states.shape}."
            )

        states = torch.from_numpy(batch.states).to(CPU_DEVICE)
        action_indices = torch.from_numpy(
            batch.action_indices
        ).to(CPU_DEVICE)
        rewards = torch.from_numpy(batch.rewards).to(CPU_DEVICE)
        next_states = torch.from_numpy(batch.next_states).to(CPU_DEVICE)
        terminals = torch.from_numpy(batch.terminals).to(CPU_DEVICE)

        self.online_network.train()

        all_current_q_values = self.online_network(states)
        current_q_values = all_current_q_values.gather(
            dim=1,
            index=action_indices.unsqueeze(1),
        ).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_network(next_states)

        targets = compute_bellman_targets(
            rewards=rewards,
            next_q_values=next_q_values,
            terminals=terminals,
            discount_factor=self.config.discount_factor,
        )

        td_errors = targets - current_q_values
        loss = F.smooth_l1_loss(
            current_q_values,
            targets,
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.online_network.parameters(),
            max_norm=self.config.gradient_clip_norm,
        )

        self.optimizer.step()
        self.update_steps += 1

        target_synchronized = (
            self.update_steps
            % self.config.target_update_interval
            == 0
        )

        if target_synchronized:
            self.synchronize_target_network()

        return TrainingResult(
            loss=float(loss.detach().item()),
            mean_abs_td_error=float(
                td_errors.detach().abs().mean().item()
            ),
            target_synchronized=target_synchronized,
        )

    def synchronize_target_network(self) -> None:
        """Copy online parameters into the frozen target network."""
        self.target_network.load_state_dict(
            self.online_network.state_dict()
        )
        self.target_network.eval()
    
    def state_dict(self) -> dict[str, Any]:
        """Export all state required to resume DQN optimization."""
        return {
            "online_network": deepcopy(
                self.online_network.state_dict()
            ),
            "target_network": deepcopy(
                self.target_network.state_dict()
            ),
            "optimizer": deepcopy(self.optimizer.state_dict()),
            "update_steps": self.update_steps,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore network, optimizer and update-counter state."""
        required_fields = {
            "online_network",
            "target_network",
            "optimizer",
            "update_steps",
        }

        if not isinstance(state, dict) or set(state) != required_fields:
            raise ValueError("Learner state has unexpected fields.")

        update_steps = state["update_steps"]

        if type(update_steps) is not int or update_steps < 0:
            raise ValueError(
                "Learner update_steps must be a non-negative integer."
            )

        try:
            self.online_network.load_state_dict(
                state["online_network"],
                strict=True,
            )
            self.target_network.load_state_dict(
                state["target_network"],
                strict=True,
            )
            self.optimizer.load_state_dict(state["optimizer"])
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            raise ValueError("Learner state is incompatible.") from error

        self.update_steps = update_steps
        self.online_network.train()
        self.target_network.eval()
        self.target_network.requires_grad_(False)

