"""Record and verify the registered Issue #85 Q-value migration contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from agent_code.DagobertDuckDQN.features import normalize_features as normalize_parent
from agent_code.DagobertDuckDQN.features import state_to_features as parent_features
from agent_code.DagobertDuckDQN.model import select_action as select_parent_action
from agent_code.DagobertDuckDQN.persistence import load_evaluation_checkpoint as load_parent
from agent_code.DagobertDuckDQNTask2.features import normalize_features as normalize_successor
from agent_code.DagobertDuckDQNTask2.features import state_to_features as successor_features
from agent_code.DagobertDuckDQNTask2.migration import INHERITED_Q_VALUE_TOLERANCE
from agent_code.DagobertDuckDQNTask2.model import select_action as select_successor_action
from agent_code.DagobertDuckDQNTask2.persistence import load_evaluation_checkpoint as load_successor

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIRECTORY = REPOSITORY_ROOT / "experiments" / "2026-09-06-dqn-task2-migration-retention"
PROBE_MANIFEST_PATH = EXPERIMENT_DIRECTORY / "probes.json"
PARENT_CHECKPOINT_PATH = REPOSITORY_ROOT / "agent_code" / "DagobertDuckDQN" / "checkpoint.pt"
CORRECTED_CHECKPOINT_PATH = (
    REPOSITORY_ROOT / "agent_code" / "DagobertDuckDQNTask2" / "checkpoint-issue85-zero-suffix.pt"
)
PARENT_SHA256 = "eb08e3f67b620ac2a253a2af4db3d5b4c6ea9e667a2aaf1d91e3fccf4ba8b05e"
CORRECTED_SHA256 = "3edb2e7196030fcb52af6c7dc9ee69d9fc1259898ea674002fe06fbe93468015"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_registered_probes(path: Path = PROBE_MANIFEST_PATH) -> tuple[int, list[dict[str, Any]]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read probe manifest: {path}") from error
    if not isinstance(data, dict) or set(data) != {"schema_version", "action_seed", "probes"}:
        raise ValueError("Probe manifest has unexpected fields.")
    if data["schema_version"] != 1 or type(data["action_seed"]) is not int:
        raise ValueError("Probe manifest schema is invalid.")
    probes = data["probes"]
    if not isinstance(probes, list) or not probes:
        raise ValueError("Probe manifest must contain probes.")
    ids = [probe.get("id") for probe in probes if isinstance(probe, dict)]
    if len(ids) != len(probes) or any(not isinstance(identifier, str) for identifier in ids):
        raise ValueError("Every probe must have a string id.")
    if len(set(ids)) != len(ids):
        raise ValueError("Probe ids must be unique.")
    return data["action_seed"], probes


def _game_state(probe: dict[str, Any]) -> dict[str, Any]:
    required = {"id", "position", "coins", "bomb_available", "crates", "bombs", "explosions"}
    if set(probe) != required:
        raise ValueError(f"Probe {probe.get('id', '<unknown>')!r} has unexpected fields.")
    field = np.zeros((7, 7), dtype=int)
    field[0, :] = -1
    field[-1, :] = -1
    field[:, 0] = -1
    field[:, -1] = -1
    explosion_map = np.zeros_like(field)
    try:
        for x, y in probe["crates"]:
            field[x, y] = 1
        for x, y in probe["explosions"]:
            explosion_map[x, y] = 1
        bombs = [(tuple(position), timer) for position, timer in probe["bombs"]]
        position = tuple(probe["position"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"Probe {probe['id']!r} has invalid geometry.") from error
    if len(position) != 2 or not isinstance(probe["bomb_available"], bool):
        raise ValueError(f"Probe {probe['id']!r} has invalid agent data.")
    return {
        "round": 1,
        "step": 1,
        "field": field,
        "self": ("probe-agent", 0, probe["bomb_available"], position),
        "coins": [tuple(coin) for coin in probe["coins"]],
        "bombs": bombs,
        "others": [],
        "explosion_map": explosion_map,
    }


def create_probe_report() -> dict[str, Any]:
    if sha256_file(PARENT_CHECKPOINT_PATH) != PARENT_SHA256:
        raise ValueError("Frozen parent checkpoint checksum mismatch.")
    if sha256_file(CORRECTED_CHECKPOINT_PATH) != CORRECTED_SHA256:
        raise ValueError("Corrected migration checkpoint checksum mismatch.")
    action_seed, probes = load_registered_probes()
    parent = load_parent(PARENT_CHECKPOINT_PATH).network
    corrected = load_successor(CORRECTED_CHECKPOINT_PATH).network
    rows: list[dict[str, Any]] = []
    for probe in probes:
        game_state = _game_state(probe)
        parent_raw = parent_features(game_state)
        successor_raw = successor_features(game_state)
        if parent_raw is None or successor_raw is None:
            raise ValueError(f"Probe {probe['id']!r} did not encode.")
        parent_state = normalize_parent(parent_raw)
        successor_state = normalize_successor(successor_raw)
        with torch.no_grad():
            parent_q = parent(torch.from_numpy(parent_state))
            successor_q = corrected(torch.from_numpy(successor_state))
        rows.append(
            {
                "id": probe["id"],
                "parent_q_values": [float(value) for value in parent_q.tolist()],
                "corrected_q_values": [float(value) for value in successor_q[:5].tolist()],
                "bomb_q_value": float(successor_q[5].item()),
                "parent_action": select_parent_action(
                    network=parent,
                    state=parent_state,
                    epsilon=0.0,
                    rng=np.random.default_rng(action_seed),
                ),
                "corrected_action": select_successor_action(
                    network=corrected,
                    state=successor_state,
                    epsilon=0.0,
                    rng=np.random.default_rng(action_seed),
                ),
            }
        )
    return {
        "schema_version": 1,
        "probe_manifest": PROBE_MANIFEST_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
        "artifact_sha256": {"parent": PARENT_SHA256, "corrected": CORRECTED_SHA256},
        "q_value_tolerance": INHERITED_Q_VALUE_TOLERANCE,
        "probes": rows,
    }


def verify_probe_report(report: dict[str, Any]) -> None:
    expected = create_probe_report()
    if not isinstance(report, dict) or set(report) != set(expected):
        raise ValueError("Probe report has unexpected fields.")
    if report["schema_version"] != 1 or report["artifact_sha256"] != expected["artifact_sha256"]:
        raise ValueError("Probe report has incompatible provenance.")
    if report["probe_manifest"] != expected["probe_manifest"]:
        raise ValueError("Probe report references the wrong manifest.")
    if report["q_value_tolerance"] != INHERITED_Q_VALUE_TOLERANCE:
        raise ValueError("Probe report has incompatible tolerance.")
    if not isinstance(report["probes"], list) or len(report["probes"]) != len(expected["probes"]):
        raise ValueError("Probe report is incomplete.")
    for actual, reference in zip(report["probes"], expected["probes"], strict=True):
        if (
            not isinstance(actual, dict)
            or set(actual) != set(reference)
            or actual["id"] != reference["id"]
        ):
            raise ValueError("Probe report has missing or substituted probes.")
        for name in ("parent_q_values", "corrected_q_values"):
            if (
                not isinstance(actual[name], list)
                or len(actual[name]) != 5
                or not np.allclose(
                    actual[name], reference[name], rtol=0.0, atol=INHERITED_Q_VALUE_TOLERANCE
                )
            ):
                raise ValueError(f"Probe {actual['id']!r} has altered {name}.")
        if not np.isclose(
            actual["bomb_q_value"],
            reference["bomb_q_value"],
            rtol=0.0,
            atol=INHERITED_Q_VALUE_TOLERANCE,
        ):
            raise ValueError(f"Probe {actual['id']!r} has an altered BOMB Q-value.")
        if (
            actual["parent_action"] != reference["parent_action"]
            or actual["corrected_action"] != reference["corrected_action"]
        ):
            raise ValueError(f"Probe {actual['id']!r} has altered actions.")
        if not np.allclose(
            actual["parent_q_values"],
            actual["corrected_q_values"],
            rtol=0.0,
            atol=INHERITED_Q_VALUE_TOLERANCE,
        ):
            raise ValueError(f"Probe {actual['id']!r} violates inherited Q-value agreement.")
        if actual["bomb_q_value"] >= max(actual["parent_q_values"]):
            raise ValueError(f"Probe {actual['id']!r} permits BOMB displacement.")
        if actual["parent_action"] != actual["corrected_action"]:
            raise ValueError(f"Probe {actual['id']!r} violates action agreement.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    report = create_probe_report()
    verify_probe_report(report)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Verified Issue #85 probe report: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
