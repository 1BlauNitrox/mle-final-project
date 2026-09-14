import json

import pytest

from scripts.analyze_task3_episode_pilot import behavioral, interval, metrics
from scripts.pilot_task3_episode_exploration import canonical_sha, config, episode_epsilon


def test_temporal_schedule_has_exact_exposure_and_independent_block_stream():
    cfg = config()
    for seed in cfg["replica_agent_seeds"]:
        eps = [episode_epsilon("episode_mixture", seed, i) for i in range(50)]
        assert sum(eps) == 10
        assert all(sum(eps[i : i + 5]) == 1 for i in range(0, 50, 5))
        assert [episode_epsilon("stepwise", seed, i) for i in range(50)] == [0.2] * 50
        assert eps == [episode_epsilon("episode_mixture", seed, i) for i in range(50)]
    with pytest.raises(ValueError):
        episode_epsilon("other", 1, 0)


def test_pilot_seed_counts_and_separation():
    cfg = config()
    train = [seed for seeds in cfg["training_world_seeds"] for seed in seeds]
    evaluation = [
        seed for suite in cfg["evaluation_suites"].values() for seed in suite["world_seeds"]
    ]
    assert len(train) == len(set(train)) == 150
    assert len(evaluation) == len(set(evaluation)) == 20
    assert not set(train) & set(evaluation)
    assert len(train) * 2 == cfg["training_episodes"]
    assert len(evaluation) * 7 * 2 == cfg["evaluation_episodes"]


def test_canonical_config_digest_is_independent_of_newlines():
    cfg = config()
    assert canonical_sha(cfg) == canonical_sha(
        json.loads(json.dumps(cfg, indent=2).replace("\n", "\r\n"))
    )


def test_repeat_check_excludes_only_timing_not_actions_or_score():
    a = {
        "agents": {
            "a": {"decision_times_ms": [1], "score": 1, "executed_action_sequence_sha256": "x"}
        }
    }
    b = {
        "agents": {
            "a": {"decision_times_ms": [2], "score": 1, "executed_action_sequence_sha256": "x"}
        }
    }
    assert behavioral(a) == behavioral(b)
    b["agents"]["a"]["executed_action_sequence_sha256"] = "y"
    assert behavioral(a) != behavioral(b)


def test_crossed_interval_preserves_replica_and_common_world_differences():
    assert interval([[2] * 5 for _ in range(3)]) == [2, 2]
    low, high = interval([[r + w for w in range(5)] for r in range(3)])
    assert low < 3 < high


def test_native_metric_mapping_and_denominator():
    own = {
        "score": 5,
        "initially_available_coins": 10,
        "coins": 2,
        "crates_destroyed": 3,
        "kills": 1,
        "survived": True,
        "survival_steps": 50,
        "self_kills": 0,
        "invalid": 0,
        "action_bomb": 4,
    }
    row = {"native": {"agents": {"DagobertDuckDQNTask3": own, "peaceful_agent": {"score": 1}}}}
    assert metrics(row)["collection_fraction"] == 0.2
    assert metrics(row)["strict_win"] == 1
    own["initially_available_coins"] = 0
    with pytest.raises(ValueError):
        metrics(row)


def test_explicit_pilot_authorization_is_required_before_compute(tmp_path, monkeypatch):
    import scripts.pilot_task3_episode_exploration as pilot

    monkeypatch.setattr(pilot, "verify", lambda root: None)
    monkeypatch.delenv("TASK168_PILOT_AUTHORIZED", raising=False)
    with pytest.raises(ValueError, match="authorization"):
        pilot.supervise(tmp_path, "training")


def test_full_synthetic_pilot_preserves_negative_retention_and_host_gates(tmp_path, monkeypatch):
    import scripts.analyze_task3_episode_pilot as analysis
    from scripts.pilot_task3_episode_exploration import sha, write, zip_json

    monkeypatch.setattr(analysis, "verify", lambda root: None)
    monkeypatch.setattr(analysis, "verify_training_tensors", lambda root: "initial")
    monkeypatch.setattr(analysis, "interval", lambda matrix: [-1, 1])
    cfg = config()
    (tmp_path / "reference.pt").write_bytes(b"reference")
    training = {"status": "completed", "completed": {}}
    model_paths = {"reference": tmp_path / "reference.pt"}
    for replica in range(3):
        for arm in cfg["arms"]:
            key = f"{arm}-r{replica + 1}"
            directory = tmp_path / "training" / key
            directory.mkdir(parents=True)
            (directory / "checkpoint.pt").write_bytes(key.encode())
            model_paths[key] = directory / "checkpoint.pt"
            rows = []
            for index, seed in enumerate(cfg["training_world_seeds"][replica]):
                own = {
                    "score": 2,
                    "initially_available_coins": 10,
                    "coins": 5,
                    "crates_destroyed": 2,
                    "kills": 0,
                    "survived": True,
                    "survival_steps": 100,
                    "self_kills": 0,
                    "invalid": 0,
                    "action_bomb": 2,
                }
                rows.append(
                    {
                        "world_seed": seed,
                        "agent_seed": cfg["replica_agent_seeds"][replica],
                        "training": True,
                        "completed_episodes": index + 1,
                        "behavior_epsilon": episode_epsilon(
                            arm, cfg["replica_agent_seeds"][replica], index
                        ),
                        "native": {"agents": {"DagobertDuckDQNTask3": own}},
                        "attack_steps": 1,
                    }
                )
            zip_json(directory / "episodes.json.gz", rows)
            write(
                directory / "result.json",
                {
                    "episodes": 50,
                    "arm": arm,
                    "replica": replica,
                    "checkpoint_sha256": sha(directory / "checkpoint.pt"),
                    "initial_online_sha256": "initial",
                    "environment": {"host": "PC"},
                    "final_online_sha256": key,
                    "optimizer_updates": 50,
                    "episodes_sha256": sha(directory / "episodes.json.gz"),
                },
            )
            training["completed"][key] = directory.relative_to(tmp_path).as_posix()
    write(tmp_path / "training-state.json", training)
    evaluation = {"status": "completed", "completed": {}}
    for artifact, path in model_paths.items():
        for suite, setting in cfg["evaluation_suites"].items():
            key = f"{artifact}-{suite}"
            directory = tmp_path / "evaluation" / key
            directory.mkdir(parents=True)
            rows = []
            for repeat in range(2):
                for index, seed in enumerate(setting["world_seeds"]):
                    kills = int(
                        index
                        < (
                            2
                            if artifact.startswith("episode")
                            else 1
                            if artifact == "reference"
                            else 0
                        )
                    )
                    own = {
                        "score": 2,
                        "initially_available_coins": 10,
                        "coins": 5,
                        "crates_destroyed": 2,
                        "kills": kills,
                        "survived": True,
                        "survival_steps": 100,
                        "self_kills": 0,
                        "invalid": 0,
                        "action_bomb": 2,
                        "decision_times_ms": [1, 2],
                    }
                    rows.append(
                        {
                            "world_seed": seed,
                            "agent_seed": seed + 1000000,
                            "repeat": repeat,
                            "suite": suite,
                            "artifact": artifact,
                            "training": False,
                            "behavior_epsilon": 0,
                            "online_before_sha256": artifact,
                            "online_after_sha256": artifact,
                            "native": {"agents": {"DagobertDuckDQNTask3": own}},
                        }
                    )
            zip_json(directory / "episodes.json.gz", rows)
            write(
                directory / "result.json",
                {
                    "artifact": artifact,
                    "suite": suite,
                    "environment": {"host": "laptop"},
                    "checkpoint_sha256": sha(path),
                    "episodes_sha256": sha(directory / "episodes.json.gz"),
                },
            )
            evaluation["completed"][key] = directory.relative_to(tmp_path).as_posix()
    write(tmp_path / "evaluation-state.json", evaluation)
    result = analysis.analyze(tmp_path)
    assert result["pilot_screen_passed"]
    assert result["selected_checkpoint"] is None
    assert not result["next_long_experiment_authorized"]

    # Round-trip the same portable transfer layout using explicitly synthetic bytes.
    import tarfile

    import scripts.pilot_task3_episode_exploration as pilot

    monkeypatch.setattr(pilot, "verify", lambda root: None)
    for name, value in (
        ("binding.json", {"tool_source": "synthetic-test"}),
        ("config.json", cfg),
        ("source-manifest.json", {}),
    ):
        write(tmp_path / name, value)
    (tmp_path / "initial.pt").write_bytes(b"synthetic-initial")
    with tarfile.open(tmp_path / "runtime-source.tar", "w"):
        pass
    transfer = tmp_path / "transfer.tar.gz"
    pilot.bundle(tmp_path, transfer)
    with pytest.raises(ValueError, match="checksum"):
        pilot.import_bundle(tmp_path / "bad-import", transfer, "0" * 64)
    assert not (tmp_path / "bad-import").exists()
    pilot.import_bundle(tmp_path / "imported", transfer, sha(transfer))
    assert set(pilot.artifacts(tmp_path / "imported")) == set(model_paths)
    # A legitimate negative result must fail the pilot, even with positive hunting.
    import gzip

    for replica in range(1, 4):
        directory = tmp_path / f"evaluation/episode_mixture-r{replica}-coin-heaven"
        with gzip.open(directory / "episodes.json.gz", "rt") as file:
            rows = json.load(file)
        for row in rows:
            row["native"]["agents"]["DagobertDuckDQNTask3"]["coins"] = 0
        zip_json(directory / "episodes.json.gz", rows)
        metadata = json.loads((directory / "result.json").read_text())
        metadata["episodes_sha256"] = sha(directory / "episodes.json.gz")
        write(directory / "result.json", metadata)
    result = analysis.analyze(tmp_path)
    assert not result["pilot_screen_passed"]
    assert not result["retention_suites"]["coin-heaven"]["coins"]
    metadata["environment"] = {"host": "different-treatment-machine"}
    write(directory / "result.json", metadata)
    with pytest.raises(ValueError, match="mixed evaluation environment"):
        analysis.analyze(tmp_path)
    metadata["environment"] = {"host": "laptop"}
    write(directory / "result.json", metadata)
    training_directory = tmp_path / "training/stepwise-r1"
    with gzip.open(training_directory / "episodes.json.gz", "rt") as file:
        training_rows = json.load(file)
    training_rows[0]["behavior_epsilon"] = 0.7
    zip_json(training_directory / "episodes.json.gz", training_rows)
    training_metadata = json.loads((training_directory / "result.json").read_text())
    training_metadata["episodes_sha256"] = sha(training_directory / "episodes.json.gz")
    write(training_directory / "result.json", training_metadata)
    with pytest.raises(ValueError, match="exploration schedule"):
        analysis.analyze(tmp_path)


def test_prepare_dispatches_one_binding_mode_without_starting_games(tmp_path, monkeypatch):
    import tarfile

    import scripts.analyze_task3_exploration as exposure
    import scripts.pilot_task3_episode_exploration as pilot

    parent = tmp_path / "parent.pt"
    parent.write_bytes(b"parent")
    original_sha = pilot.sha

    def digest(path):
        if path == parent:
            return pilot.INPUT
        if path.name == "phase-c-evidence.tar.gz":
            return config()["phase_c_evidence_sha256"]
        return original_sha(path)

    monkeypatch.setattr(pilot, "sha", digest)
    monkeypatch.setattr(pilot, "verify", lambda root: None)
    monkeypatch.setattr(
        exposure, "analyze_input", lambda path: {"decision": {"learning_pilot_ready": True}}
    )
    monkeypatch.setattr(
        pilot.subprocess,
        "check_output",
        lambda command, **kwargs: "" if "status" in command else "commit",
    )
    modes = []

    def run(command, **kwargs):
        if command[0] == "git":
            destination = next(
                argument.split("=", 1)[1]
                for argument in command
                if argument.startswith("--output=")
            )
            with tarfile.open(destination, "w"):
                pass
        else:
            assert command[2:] == ["_bind", "--root", str(tmp_path / "prepared")]
            modes.append(command[2])
            (tmp_path / "prepared/initial.pt").write_bytes(b"bound")

    monkeypatch.setattr(pilot.subprocess, "run", run)
    pilot.prepare(tmp_path / "prepared", parent)
    assert modes == ["_bind"]


def test_raw_tensor_counters_must_match_training_evidence(tmp_path):
    import hashlib

    import torch

    from scripts.analyze_task3_episode_pilot import verify_training_tensors
    from scripts.pilot_task3_episode_exploration import write

    state = {"weight": torch.tensor([1.0])}
    digest = hashlib.sha256(b"weight" + state["weight"].numpy().tobytes()).hexdigest()
    initial = {
        "learner_state": {"online_network": state},
        "config": {},
        "actions": [],
        "rewards": {},
        "checkpoint_schema_version": 1,
        "model_schema_version": 1,
        "feature_schema_version": 1,
    }
    torch.save(initial, tmp_path / "initial.pt")
    directory = tmp_path / "job"
    directory.mkdir()
    payload = {
        **initial,
        "completed_episodes": 50,
        "agent_seed": 2683001,
        "learner_state": {"online_network": state, "update_steps": 10},
    }
    torch.save(payload, directory / "checkpoint.pt")
    write(
        directory / "result.json",
        {"replica": 0, "optimizer_updates": 10, "final_online_sha256": digest},
    )
    write(tmp_path / "training-state.json", {"completed": {"job": "job"}})
    assert verify_training_tensors(tmp_path) == digest
    payload["completed_episodes"] = 49
    torch.save(payload, directory / "checkpoint.pt")
    with pytest.raises(ValueError, match="counters"):
        verify_training_tensors(tmp_path)


def test_resume_rejects_changed_host_before_any_worker(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import psutil

    import scripts.pilot_task3_episode_exploration as pilot

    monkeypatch.setattr(pilot, "verify", lambda root: None)
    monkeypatch.setenv("TASK168_PILOT_AUTHORIZED", "yes")
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=10**12))
    monkeypatch.setattr(pilot.shutil, "disk_usage", lambda root: SimpleNamespace(free=10**12))
    pilot.write(
        tmp_path / "training-state.json", {"environment": {"host": "different"}, "status": "failed"}
    )
    with pytest.raises(ValueError, match="environment changed"):
        pilot.supervise(tmp_path, "training", resume=True)
