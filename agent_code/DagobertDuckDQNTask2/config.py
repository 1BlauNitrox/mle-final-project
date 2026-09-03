"""Central configuration for the DQN Task 2 successor.

Hyperparameter revision (issue #44): the Task 1 defaults are not reused
verbatim. `DagobertDuckDQN`'s `epsilon_decay=0.99` over a 10,000-episode
training budget reaches its `minimum_epsilon` floor after about 230 episodes
(`0.99**230 ~= 0.1`) -- under 3% of the run. Issue #58 found training reached
0.989-1.000 mean coin collection while deterministic greedy evaluation of the
same checkpoints only reached 0.65-0.92: the policy was trained almost
entirely at minimum exploration and never had to cope with its own greedy
mistakes during the other 97%. `epsilon_decay=0.9997` reaches the same floor
around episode 8,000 instead, spending most of the budget exploring. This is
an implementation default for Task 2, not a validated fix -- it has not been
tested as a controlled variable and must not be cited as evidence until it
is (mirrors the caveat issue #58 applies to its own reward-shaping values).

`learning_rate` and `target_update_interval` are also revised, conservatively,
for the larger Task 2 input/action space (21 features and 6 actions versus 8
and 5): a smaller learning rate and less frequent target synchronization
both reduce how much a single noisy update can move the value estimates.
`replay_warmup` is raised so training does not start learning from as narrow
a slice of experience. None of these three carry the same specific,
computed rationale as the epsilon schedule; they are documented, deliberate,
and still implementation defaults pending a real experiment.
"""

from __future__ import annotations

from dataclasses import dataclass

ACTIONS: tuple[str, ...] = (
    "UP",
    "RIGHT",
    "DOWN",
    "LEFT",
    "WAIT",
    "BOMB",
)

ACTION_TO_INDEX: dict[str, int] = {
    action: index for index, action in enumerate(ACTIONS)
}

FEATURE_COUNT = 21
FEATURE_SCHEMA_VERSION = 2

REWARDS: dict[str, float] = {
    "COIN_COLLECTED": 10.0,
    "INVALID_ACTION": -0.5,
    "WAITED": -0.1,
    # Inherited from the Task 1 lineage (issue #58); not potential-based.
    "MOVED_TOWARDS_COIN": 0.1,
    "MOVED_AWAY_FROM_COIN": -0.1,
    # Task 2 additions (issue #44). All native framework events (events.py);
    # no custom derivation is required for these five.
    "CRATE_DESTROYED": 1.0,
    "COIN_FOUND": 2.0,
    "KILLED_SELF": -10.0,
    "GOT_KILLED": -10.0,
    "SURVIVED_ROUND": 5.0,
    # Custom shaping derived from the pre-action state (train.py), rewarding
    # BOMB only when it will actually destroy a crate. BOMB_DROPPED and
    # BOMB_EXPLODED are deliberately left unrewarded so bomb placement is
    # learned from its consequences, not from a flat per-placement bonus
    # that would encourage spamming bombs regardless of target.
    "USEFUL_BOMB_PLACED": 0.5,
    "WASTEFUL_BOMB_PLACED": -0.5,
}


@dataclass(frozen=True)
class DQNConfig:
    """Validated architecture and training defaults for the DQN"""

    input_dim: int = FEATURE_COUNT
    hidden_sizes: tuple[int, ...] = (64, 64)
    output_dim: int = len(ACTIONS)

    learning_rate: float = 0.0005
    discount_factor: float = 0.9
    gradient_clip_norm: float = 10.0

    batch_size: int = 64
    replay_capacity: int = 10_000
    replay_warmup: int = 500
    target_update_interval: int = 500

    initial_epsilon: float = 1.0
    epsilon_decay: float = 0.9997
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
