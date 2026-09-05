"""Prepare seed-compatible starting checkpoints for Issue #46."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from agent_code.DagobertDuckDQNTask2.persistence import (
    CHECKPOINT_PATH,
    load_training_checkpoint,
    save_checkpoint,
)
from agent_code.DagobertDuckDQNTask2.replay import ReplayBuffer
from training.run_experiment import REPOSITORY_ROOT

ISSUE = 46
SOURCE_SHA256 = "44cd337001b27b8596eaed985cfae1d7f30ecaf0b6b0328b35185395b7b81b6e"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "training_outputs" / "issue-46-starting-artifacts"


@dataclass(frozen=True)
class ReplicaSeed:
    replica: str
    world_seed: int
    agent_seed: int


REPLICAS = (
    ReplicaSeed("r1", 14_001, 24_001),
    ReplicaSeed("r2", 14_002, 24_002),
    ReplicaSeed("r3", 14_003, 24_003),
    ReplicaSeed("r4", 14_004, 24_004),
    ReplicaSeed("r5", 14_005, 24_005),
)


def prepare_starting_artifacts(
    output_directory: Path = DEFAULT_OUTPUT,
    source_checkpoint: Path = CHECKPOINT_PATH,
) -> Path:
    """Create five starts with equal weights and independent RNG streams."""
    output_directory = Path(output_directory).resolve()
    source_checkpoint = Path(source_checkpoint).resolve()
    manifest_path = output_directory / "manifest.json"

    source_hash = _sha256(source_checkpoint)
    if source_hash != SOURCE_SHA256:
        raise ValueError(
            f"Issue #46 source checkpoint mismatch: expected {SOURCE_SHA256}, got {source_hash}"
        )

    if manifest_path.exists():
        _verify_existing(
            manifest_path,
            output_directory,
            source_checkpoint,
            source_hash,
        )
        return manifest_path
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(
            f"Refusing to replace incomplete output directory: {output_directory}"
        )

    source = load_training_checkpoint(source_checkpoint)
    if source.completed_episodes != 0 or len(source.replay_buffer) != 0:
        raise ValueError("Issue #46 source must be an untrained empty-replay checkpoint")
    if source.learner.update_steps != 0 or source.epsilon != source.config.initial_epsilon:
        raise ValueError("Issue #46 source contains non-fresh optimizer or epsilon state")

    output_directory.mkdir(parents=True, exist_ok=True)
    artifact_records: list[dict[str, object]] = []
    for replica in REPLICAS:
        action_rng, replay_seed = _initial_random_streams(replica.agent_seed)
        replay_buffer = ReplayBuffer(
            capacity=source.config.replay_capacity,
            seed=replay_seed,
        )
        path = output_directory / replica.replica / "checkpoint.pt"
        save_checkpoint(
            learner=source.learner,
            replay_buffer=replay_buffer,
            action_rng=action_rng,
            epsilon=source.config.initial_epsilon,
            completed_episodes=0,
            agent_seed=replica.agent_seed,
            path=path,
        )
        _verify_artifact(path, source_checkpoint, replica.agent_seed)
        artifact_records.append(
            {
                **asdict(replica),
                "path": _display_path(path),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )

    manifest = {
        "schema_version": 1,
        "issue": ISSUE,
        "source": {
            "path": _display_path(source_checkpoint),
            "sha256": source_hash,
        },
        "invariant": (
            "All online/target weights and optimizer state equal the reviewed "
            "migration; only the registered action/replay RNG streams differ."
        ),
        "artifacts": artifact_records,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _initial_random_streams(agent_seed: int) -> tuple[np.random.Generator, int]:
    root = np.random.SeedSequence(agent_seed)
    action_sequence, replay_sequence = root.spawn(2)
    action_rng = np.random.default_rng(action_sequence)
    replay_seed = int(replay_sequence.generate_state(1, dtype=np.uint64)[0])
    return action_rng, replay_seed


def _verify_artifact(path: Path, source_path: Path, expected_seed: int) -> None:
    candidate = load_training_checkpoint(path)
    source = load_training_checkpoint(source_path)
    if candidate.agent_seed != expected_seed:
        raise ValueError(f"Unexpected agent seed in {path}")
    if candidate.completed_episodes != 0 or len(candidate.replay_buffer) != 0:
        raise ValueError(f"Starting artifact is not fresh: {path}")
    if candidate.learner.update_steps != 0:
        raise ValueError(f"Starting artifact has optimizer updates: {path}")
    for network_name in ("online_network", "target_network"):
        candidate_state = getattr(candidate.learner, network_name).state_dict()
        source_state = getattr(source.learner, network_name).state_dict()
        if any(not torch.equal(candidate_state[name], source_state[name]) for name in source_state):
            raise ValueError(f"Starting weights differ from migration source: {path}")
    if not _equal_state(
        candidate.learner.optimizer.state_dict(),
        source.learner.optimizer.state_dict(),
    ):
        raise ValueError(f"Starting optimizer state differs from migration source: {path}")


def _verify_existing(
    manifest_path: Path,
    output_directory: Path,
    source_checkpoint: Path,
    source_hash: str,
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("issue") != ISSUE or manifest.get("source", {}).get("sha256") != source_hash:
        raise ValueError("Existing Issue #46 starting-artifact manifest is incompatible")
    records = manifest.get("artifacts")
    if not isinstance(records, list) or len(records) != len(REPLICAS):
        raise ValueError("Existing Issue #46 starting-artifact manifest is incomplete")
    by_replica = {record.get("replica"): record for record in records}
    for replica in REPLICAS:
        record = by_replica.get(replica.replica)
        path = output_directory / replica.replica / "checkpoint.pt"
        if record is None or not path.is_file() or record.get("sha256") != _sha256(path):
            raise ValueError(f"Existing starting artifact is missing or changed: {replica.replica}")
        _verify_artifact(path, source_checkpoint, replica.agent_seed)


def _equal_state(left: object, right: object) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _equal_state(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _equal_state(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    return left == right


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return str(path)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    try:
        manifest = prepare_starting_artifacts(arguments.output)
    except Exception as error:
        print(f"Preparation failed: {error}", file=sys.stderr)
        return 1
    print(f"Issue #46 starting artifacts ready: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
