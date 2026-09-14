"""Prepare #137 only from the verified, mechanically selected peaceful result."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_code.DagobertDuckDQNTask3.model import DQNLearner  # noqa: E402
from agent_code.DagobertDuckDQNTask3.persistence import (  # noqa: E402
    load_training_checkpoint,
    save_checkpoint,
)
from agent_code.DagobertDuckDQNTask3.replay import ReplayBuffer  # noqa: E402
from training.analyze_task3_campaign import verify_compact  # noqa: E402
from training.run_task3_campaign import (  # noqa: E402
    protocol_path,
    read_json,
    require,
    sha256,
)


def prepare(parent: Path, peaceful_evidence: Path, output: Path):
    parent, peaceful_evidence, output = (
        parent.resolve(),
        peaceful_evidence.resolve(),
        output.resolve(),
    )
    require(not output.exists(), "Output already exists; preserve previous bindings")
    require(
        verify_compact(peaceful_evidence)["status"] == "exploratory_pass",
        "Coin-collector requires a passing peaceful result",
    )
    result = read_json(peaceful_evidence / "result.json")
    require(
        sha256(parent) == result.get("selected_artifact_sha256"),
        "Parent is not the mechanically selected peaceful checkpoint",
    )
    loaded = load_training_checkpoint(parent)
    require(
        loaded.completed_episodes == 10000, "Parent must be the exact final peaceful checkpoint"
    )
    learner = DQNLearner(config=loaded.config, seed=44)
    learner.online_network.load_state_dict(loaded.learner.online_network.state_dict())
    learner.target_network.load_state_dict(loaded.learner.target_network.state_dict())
    output.mkdir(parents=True)
    frozen = output / "peaceful-parent.pt"
    successor = output / "coincollector-start.pt"
    shutil.copyfile(parent, frozen)
    save_checkpoint(
        learner=learner,
        replay_buffer=ReplayBuffer(capacity=loaded.config.replay_capacity, seed=44),
        action_rng=np.random.default_rng(44),
        epsilon=loaded.config.initial_epsilon,
        completed_episodes=0,
        agent_seed=44,
        path=successor,
    )
    evidence_copy = output / "peaceful-evidence"
    evidence_copy.mkdir()
    for name in ("result.json", "observations.json.gz"):
        shutil.copyfile(peaceful_evidence / name, evidence_copy / name)
    modes = {
        "action_masking": "framework_legal" if loaded.config.action_masking else "none",
        "escape_continuations": "on" if loaded.config.escape_continuation_features else "off",
    }
    config = yaml.safe_load(protocol_path("coincollector").read_text(encoding="utf-8"))
    plans = {}
    for name, source in config["plans"].items():
        data = yaml.safe_load((ROOT / source).read_text(encoding="utf-8"))
        data.update(modes)
        data["plan_id"] = "issue137-exploratory-" + name
        for replica in data["replicas"]:
            replica["parent_artifact"] = str(successor if name == "candidate" else frozen)
        path = output / (name + ".yaml")
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        plans[name] = {"path": path.name, "sha256": sha256(path)}
    binding = {
        "schema_version": 1,
        "status": "exploratory_coincollector_pending_review",
        "scientific_training_authorized": False,
        "task2_complete": False,
        "parent": {
            "path": frozen.name,
            "sha256": sha256(frozen),
            "size_bytes": frozen.stat().st_size,
            "source_commit": result["authorization"]["identity"]["reviewed_commit"],
        },
        "successor": {
            "path": successor.name,
            "sha256": sha256(successor),
            "size_bytes": successor.stat().st_size,
            "input_dim": 39,
        },
        "prerequisite": {
            "path": evidence_copy.name,
            "result_sha256": sha256(evidence_copy / "result.json"),
            "selected_replica": result["selected_replica"],
        },
        "modes": modes,
        "plans": plans,
    }
    (output / "binding.json").write_text(json.dumps(binding, indent=2) + "\n", encoding="utf-8")
    return binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--peaceful-evidence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.parent, args.peaceful_evidence, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
