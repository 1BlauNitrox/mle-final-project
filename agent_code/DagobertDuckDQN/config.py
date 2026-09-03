"""Central configuration for the DagobertDuckDQN agent"""

from __future__ import annotations

from dataclasses import dataclass

ACTIONS: tuple[str, ...] = (
    "UP",
    "RIGHT",
    "DOWN",
    "LEFT",
    "WAIT",
)

ACTION_TO_INDEX: dict[str, int] = {
    action: index for index, action in enumerate(ACTIONS)
}

FEATURE_COUNT = 8
FEATURE_SCHEMA_VERSION = 1

REWARDS: dict[str, float] = {
    "COIN_COLLECTED": 10.0,
    "INVALID_ACTION": -0.5,
    "WAITED": -0.1,
    # Custom shaping events emitted by train.py, matching the definition and
    # magnitudes validated on the tabular agent. Not potential-based, so policy
    # invariance is not guaranteed; see the issue #58 experiment record.
    "MOVED_TOWARDS_COIN": 0.1,
    "MOVED_AWAY_FROM_COIN": -0.1,
}

@dataclass(frozen=True)
class DQNConfig:
    """Validated architecture and training defaults for the DQN"""

    input_dim: int = FEATURE_COUNT
    hidden_sizes: tuple[int, ...] = (64, 64)
    output_dim: int = len(ACTIONS)

    learning_rate: float = 0.001
    discount_factor: float = 0.9
    gradient_clip_norm: float = 10.0

    batch_size: int = 64
    replay_capacity: int = 10_000
    replay_warmup: int = 256
    target_update_interval: int = 250

    initial_epsilon: float = 1.0
    epsilon_decay: float = 0.99
    minimum_epsilon: float = 0.1

    default_seed: int = 0
    torch_num_threads: int = 1

    def __post_init__(self) -> None:
        """Reject internally inconsistent configurations."""
        if self.input_dim != FEATURE_COUNT:
            raise ValueError("input_dim must match FEATURE_COUNT.")

        if self.output_dim != len(ACTIONS):
            raise ValueError("output_dim must match the action count.")

        if not self.hidden_sizes or any(size <= 0 for size in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain positive layer sizes.")

        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")

        if not 0.0 <= self.discount_factor <= 1.0:
            raise ValueError("discount_factor must be in [0, 1].")

        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive.")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        if self.replay_capacity < self.batch_size:
            raise ValueError("replay_capacity must be at least batch_size.")

        if not self.batch_size <= self.replay_warmup <= self.replay_capacity:
            raise ValueError(
                "replay_warmup must be between batch_size and replay_capacity."
            )

        if self.target_update_interval <= 0:
            raise ValueError("target_update_interval must be positive.")

        if not 0.0 <= self.minimum_epsilon <= self.initial_epsilon <= 1.0:
            raise ValueError("The epsilon bounds are inconsistent.")

        if not 0.0 < self.epsilon_decay <= 1.0:
            raise ValueError("epsilon_decay must be in (0, 1].")

        if self.torch_num_threads != 1:
            raise ValueError("torch_num_threads must be exactly one.")


DEFAULT_CONFIG = DQNConfig()
