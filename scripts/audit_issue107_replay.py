"""Audit retained protected replay snapshots without running training or games."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import tempfile
from pathlib import Path

import numpy as np

from agent_code.DagobertDuckDQNTask2.persistence import load_training_checkpoint
from agent_code.DagobertDuckDQNTask2.replay import ReplayBuffer

ARCHIVE_SHA256 = "ef2c0fac3db0e5b72a9ea9ce89599f726a495a7a84ba33bb9d9b3f4342fc5748"


def partition_digest(partition):
    result = hashlib.sha256()
    for name, array in sorted(partition.items()):
        result.update(name.encode() + str(array.dtype).encode())
        result.update(str(array.shape).encode() + array.tobytes())
    return result.hexdigest()


def audit(archive_path):
    with archive_path.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != ARCHIVE_SHA256:
            raise ValueError("Not the registered Issue #107 evidence archive")
    observations = []
    with tarfile.open(archive_path, "r:gz") as archive, tempfile.TemporaryDirectory() as temporary:
        for cell, plan in (("C", "replay"), ("D", "combined")):
            for replica in range(1, 6):
                protected_hash = None
                previous_updates = -1
                for stage, episodes in (
                    ("visible-coins", 2000),
                    ("dense-crates", 4000),
                    ("classic-crates", 10000),
                ):
                    name = (
                        f"run-plans/issue107-cell-{cell.lower()}-{plan}/"
                        f"artifacts/r{replica}/{stage}/checkpoint.pt"
                    )
                    member = archive.getmember(name)
                    if not member.isfile():
                        raise ValueError("Expected regular checkpoint file")
                    checkpoint = Path(temporary) / "checkpoint.pt"
                    checkpoint.write_bytes(archive.extractfile(member).read())
                    loaded = load_training_checkpoint(checkpoint)
                    replay = loaded.replay_buffer
                    state = replay.state_dict()
                    current_hash = partition_digest(state["protected"])
                    if protected_hash is None:
                        protected_hash = current_hash
                    clone = ReplayBuffer(capacity=10000, seed=0, mode="protected_task1")
                    clone.load_state_dict(state)
                    first, second = replay.sample(64), clone.sample(64)
                    checks = {
                        "protected_size_2000": replay.protected_size == 2000,
                        "collection_closed": not replay.collection_open,
                        "protected_contents_preserved": current_hash == protected_hash,
                        "episode_counter": loaded.completed_episodes == episodes,
                        "updates_monotonic": loaded.learner.update_steps >= previous_updates,
                        "epsilon_schedule": bool(
                            np.isclose(
                                loaded.epsilon,
                                max(
                                    loaded.config.minimum_epsilon,
                                    loaded.config.initial_epsilon
                                    * loaded.config.epsilon_decay**episodes,
                                ),
                            )
                        ),
                        "restored_sampling_matches": all(
                            np.array_equal(getattr(first, key), getattr(second, key))
                            for key in vars(first)
                        ),
                        # The registered quota backfills from protected replay
                        # until 48 other transitions are available.
                        "registered_quota": int(first.protected_flags.sum())
                        == 64 - min(48, replay.other_size),
                    }
                    observations.append(
                        {
                            "cell": cell,
                            "replica": f"r{replica}",
                            "stage": stage,
                            "artifact": name,
                            "protected_sha256": current_hash,
                            "updates": loaded.learner.update_steps,
                            "epsilon": loaded.epsilon,
                            "other_size": replay.other_size,
                            "sampled_protected": int(first.protected_flags.sum()),
                            "checks": checks,
                        }
                    )
                    previous_updates = loaded.learner.update_steps
    return {
        "archive_sha256": ARCHIVE_SHA256,
        "snapshots": observations,
        "passed": all(all(row["checks"].values()) for row in observations),
        "limitation": (
            "Snapshot audit and current sampling replay do not reconstruct historical "
            "minibatches or prove optimizer continuity between every episode."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"snapshots": len(result["snapshots"]), "passed": result["passed"]}))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
