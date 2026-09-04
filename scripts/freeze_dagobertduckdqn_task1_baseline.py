"""Export the frozen Task 1 DagobertDuckDQN artifact selected under issue #42.

Issue #58's registered stopping rule ("no third Task 1 DQN tuning experiment")
and the prospective candidate-selection rule posted to #42 together identify
`run-02` of the #58 series as the candidate. This script verifies that the
given `--source` file is genuinely that run's retained checkpoint (fixed
checksum, size, seed, and episode count -- not just seed/episode-count alone,
since those two values are not unique to a single specific file) and exports
an evaluation-only artifact: the online network's weights plus the schema and
configuration evaluation needs, and nothing a resumable training checkpoint
would additionally carry (target network, optimizer, replay buffer, RNG
state, epsilon, agent seed). Reviewers flagged an earlier version of this
script for producing a "compact" artifact that reset those fields to empty
rather than omitting them -- this version omits them, matching the
evaluation-only schema `persistence.py` now defines.

Run once, locally, against the retained training output. The result is what
gets committed; this script is kept for reviewer transparency and
reproducibility, not for repeated execution.
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
from pathlib import Path
from typing import Any

import torch

from agent_code.DagobertDuckDQN.config import ACTIONS, FEATURE_SCHEMA_VERSION, REWARDS, DQNConfig
from agent_code.DagobertDuckDQN.model import CPU_DEVICE, build_q_network
from agent_code.DagobertDuckDQN.persistence import (
    CHECKPOINT_PATH,
    MODEL_SCHEMA_VERSION,
    _restore_config,
    load_evaluation_checkpoint,
    save_evaluation_artifact,
)

# The retained checkpoint this script must be pointed at: run-02 of issue
# #58's series (training_outputs/issue-58-dqn-task1-movement-shaping/
# 20260902T213535561904Z/artifacts/run-02-final-checkpoint.pt). That path is
# not committed (raw training output stays outside Git), so the checksum
# below -- not a recorded path -- is what proves a given `--source` file is
# genuinely this run and not merely a file that happens to share its seed
# and episode count.
EXPECTED_SOURCE_SHA256 = "1d8f2bc9c33d775b59595f0b5ae0978078e2f2fe3571a9a1299748a042857924"
EXPECTED_SOURCE_SIZE_BYTES = 862842
EXPECTED_WORLD_SEED = 15002
EXPECTED_AGENT_SEED = 25002
EXPECTED_COMPLETED_EPISODES = 10_000

# The source checkpoint's own (pre-freeze) schema: a resumable training
# checkpoint, matching what DagobertDuckDQN/persistence.py wrote before this
# freeze and what DagobertDuckDQNTask2/persistence.py (issue #43) still
# writes today. Duplicated here, deliberately: persistence.py now defines
# only the frozen agent's evaluation-only schema and must not regain a
# resumable-checkpoint reader just to support this one-off export.
_SOURCE_REQUIRED_FIELDS = {
    "checkpoint_schema_version",
    "model_schema_version",
    "feature_schema_version",
    "actions",
    "rewards",
    "config",
    "agent_seed",
    "epsilon",
    "completed_episodes",
    "learner_state",
    "replay_state",
    "action_rng_state",
}
_SOURCE_LEARNER_REQUIRED_FIELDS = {
    "online_network",
    "target_network",
    "optimizer",
    "update_steps",
}


def sha256_file(path: Path) -> str:
    """Calculate the SHA-256 checksum of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def verify_source_provenance(source: Path) -> None:
    """Reject any file that is not run-02's retained checkpoint, byte for byte."""
    if not source.is_file():
        raise FileNotFoundError(f"Source checkpoint does not exist: {source}")

    actual_size = source.stat().st_size
    if actual_size != EXPECTED_SOURCE_SIZE_BYTES:
        raise ValueError(
            f"Source checkpoint size {actual_size} does not match run-02's "
            f"recorded size {EXPECTED_SOURCE_SIZE_BYTES}."
        )

    actual_sha256 = sha256_file(source)
    if actual_sha256 != EXPECTED_SOURCE_SHA256:
        raise ValueError(
            f"Source checkpoint checksum {actual_sha256} does not match "
            f"run-02's recorded checksum {EXPECTED_SOURCE_SHA256}. Refusing "
            "to export from a file that is not verifiably run-02."
        )


def load_source_online_network_state(
    source: Path,
) -> tuple[dict[str, Any], DQNConfig, int]:
    """Read just enough of the source resumable checkpoint to export it.

    Applies the same restricted deserialization and schema strictness the
    removed resumable loader had, but returns only the online network's raw
    state dict, the reconstructed config, and the completed-episode count --
    everything else in the source (replay, optimizer, RNG state, epsilon,
    agent seed) is deliberately not carried into the frozen artifact.
    """
    try:
        payload = torch.load(source, map_location=CPU_DEVICE, weights_only=True)
    except (
        EOFError,
        OSError,
        pickle.UnpicklingError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        raise ValueError(f"Could not load source checkpoint: {source}") from error

    if not isinstance(payload, dict) or set(payload) != _SOURCE_REQUIRED_FIELDS:
        raise ValueError("Source checkpoint has unexpected fields.")

    if payload["model_schema_version"] != MODEL_SCHEMA_VERSION:
        raise ValueError("Source model schema version mismatch.")

    if payload["feature_schema_version"] != FEATURE_SCHEMA_VERSION:
        raise ValueError("Source feature schema version mismatch.")

    if payload["actions"] != list(ACTIONS):
        raise ValueError("Source action order mismatch.")

    if payload["rewards"] != REWARDS:
        raise ValueError("Source reward mapping mismatch.")

    agent_seed = payload["agent_seed"]
    completed_episodes = payload["completed_episodes"]

    if agent_seed != EXPECTED_AGENT_SEED:
        raise ValueError(
            f"Source agent_seed {agent_seed} does not match the expected "
            f"run-02 agent_seed {EXPECTED_AGENT_SEED}."
        )

    if completed_episodes != EXPECTED_COMPLETED_EPISODES:
        raise ValueError(
            f"Source checkpoint completed {completed_episodes} episodes, "
            f"expected {EXPECTED_COMPLETED_EPISODES}."
        )

    config = _restore_config(payload["config"])

    learner_state = payload["learner_state"]
    if not isinstance(learner_state, dict) or set(learner_state) != _SOURCE_LEARNER_REQUIRED_FIELDS:
        raise ValueError("Source learner state has unexpected fields.")

    return learner_state["online_network"], config, completed_episodes


def verify_policy_preserved(
    source_network_state: dict[str, Any],
    frozen: Path,
    config: DQNConfig,
) -> None:
    """Confirm the frozen artifact selects identically to the source checkpoint."""
    source_network = build_q_network(config, seed=0)
    source_network.load_state_dict(source_network_state, strict=True)

    frozen_network = load_evaluation_checkpoint(frozen).network

    probe_states = torch.randn(64, config.input_dim)

    with torch.no_grad():
        source_values = source_network(probe_states)
        frozen_values = frozen_network(probe_states)

    if not torch.equal(source_values, frozen_values):
        raise RuntimeError(
            "Frozen artifact does not reproduce the source network's Q-values."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to run-02's retained full checkpoint "
        "(training_outputs/issue-58-.../artifacts/run-02-final-checkpoint.pt).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=CHECKPOINT_PATH,
        help="Destination for the frozen, evaluation-only artifact.",
    )
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing frozen artifact: {args.output}"
        )

    verify_source_provenance(args.source)

    online_network_state, config, completed_episodes = load_source_online_network_state(
        args.source
    )

    network = build_q_network(config, seed=0)
    network.load_state_dict(online_network_state, strict=True)

    save_evaluation_artifact(
        network=network,
        config=config,
        completed_episodes=completed_episodes,
        path=args.output,
    )

    verify_policy_preserved(online_network_state, args.output, config)

    source_size = args.source.stat().st_size
    frozen_size = args.output.stat().st_size

    print(f"Source checkpoint:  {args.source} ({source_size} bytes)")
    print(f"Frozen artifact:    {args.output} ({frozen_size} bytes)")
    print(f"Frozen SHA-256:     {sha256_file(args.output)}")
    print(f"Reduction:          {source_size - frozen_size} bytes")


if __name__ == "__main__":
    main()
