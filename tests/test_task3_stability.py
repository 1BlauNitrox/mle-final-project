import copy
import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from scripts.analyze_task3_stability import behavioral, interval  # noqa: E402
from scripts.frozen_opponent_inputs import FrozenOpponentInputs, validate_frozen  # noqa: E402
from scripts.pilot_task3_stability import config, episode_epsilon  # noqa: E402


def learner():
    torch.set_num_threads(1)
    torch.manual_seed(173)
    network = torch.nn.Module()
    network.layers = torch.nn.Sequential(
        torch.nn.Linear(39, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 6),
    )
    return SimpleNamespace(
        online_network=network,
        optimizer=torch.optim.Adam(network.parameters(), lr=0.0005),
        target_network=copy.deepcopy(network),
    )


def update(subject):
    torch.manual_seed(42)
    inputs = torch.randn(64, 39)
    subject.optimizer.zero_grad(set_to_none=True)
    torch.nn.functional.smooth_l1_loss(
        subject.online_network.layers(inputs), torch.ones(64, 6)
    ).backward()
    torch.nn.utils.clip_grad_norm_(subject.online_network.parameters(), 10)
    subject.optimizer.step()


def test_frozen_weights_q_equivalence_and_checkpoint_resume(tmp_path):
    subject = learner()
    anchor = copy.deepcopy(subject.online_network.state_dict())
    guard = FrozenOpponentInputs(subject, anchor)
    x = torch.randn(100, 39)
    x[:, 26:] = 0
    before = subject.online_network.layers(x).detach().clone()
    for _ in range(3):
        update(subject)
    guard.validate()
    assert torch.equal(before, subject.online_network.layers(x))
    assert not torch.equal(
        anchor["layers.0.weight"][:, 26:],
        subject.online_network.state_dict()["layers.0.weight"][:, 26:],
    )
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "network": subject.online_network.state_dict(),
            "optimizer": subject.optimizer.state_dict(),
        },
        path,
    )
    restored = learner()
    state = torch.load(path, weights_only=True)
    restored.online_network.load_state_dict(state["network"])
    restored.optimizer.load_state_dict(state["optimizer"])
    restored_guard = FrozenOpponentInputs(restored, anchor)
    update(restored)
    update(subject)
    restored_guard.validate()
    assert all(
        torch.equal(v, restored.online_network.state_dict()[k])
        for k, v in subject.online_network.state_dict().items()
    )
    # Target remains free to follow the normal hard-sync schedule.
    restored.target_network.load_state_dict(restored.online_network.state_dict())
    assert all(
        torch.equal(v, restored.target_network.state_dict()[k])
        for k, v in restored.online_network.state_dict().items()
    )


def test_guard_rejects_frozen_momentum_drift_and_nonfinite():
    subject = learner()
    anchor = copy.deepcopy(subject.online_network.state_dict())
    guard = FrozenOpponentInputs(subject, anchor)
    update(subject)
    first = dict(subject.online_network.named_parameters())["layers.0.weight"]
    subject.optimizer.state[first]["exp_avg"][0, 0] = 1
    with pytest.raises(ValueError, match="Adam moment"):
        guard.validate()
    subject.optimizer.state[first]["exp_avg"][0, 0] = 0
    with torch.no_grad():
        first[0, 0] += 1
    with pytest.raises(ValueError, match="moved"):
        guard.validate()
    with torch.no_grad():
        first[0, 0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        validate_frozen(subject.online_network.state_dict(), anchor)


def test_control_really_updates_inherited_weights():
    subject = learner()
    anchor = copy.deepcopy(subject.online_network.state_dict())
    update(subject)
    with pytest.raises(ValueError, match="moved"):
        validate_frozen(subject.online_network.state_dict(), anchor)


def test_registered_matrix_exploration_and_uncertainty():
    cfg = config()
    train = [s for group in cfg["training_world_seeds"] for s in group]
    dev = [s for suite in cfg["evaluation_suites"].values() for s in suite["world_seeds"]]
    assert len(train) == len(set(train)) == 500
    assert len(dev) == len(set(dev)) == 160
    assert not set(train) & set(dev)
    assert len(train) * 2 == cfg["training_episodes"] == 1000
    assert len(dev) * 11 * 2 == cfg["evaluation_episodes"] == 3520
    for seed in cfg["replica_agent_seeds"]:
        eps = [episode_epsilon("treatment", seed, i) for i in range(100)]
        assert eps == [episode_epsilon("control", seed, i) for i in range(100)]
        assert sum(eps) == 20
        assert all(sum(eps[i : i + 5]) == 1 for i in range(0, 100, 5))
    assert interval([[0.2] * 40 for _ in range(5)]) == pytest.approx([0.2, 0.2])
    assert behavioral({"score": 1, "decision_times_ms": [2]}) == {"score": 1}
    assert behavioral({"score": 2}) != behavioral({"score": 1})


def test_full_synthetic_pilot_preserves_negative_retention_and_host_gates(tmp_path, monkeypatch):
    import scripts.analyze_task3_stability as analysis
    from scripts.pilot_task3_stability import sha, write, zip_json

    monkeypatch.setattr(analysis, "verify", lambda root: None)
    monkeypatch.setattr(analysis, "verify_training_tensors", lambda root: "initial")
    monkeypatch.setattr(analysis, "interval", lambda matrix: [0.01, 0.3])
    cfg = config()
    (tmp_path / "reference.pt").write_bytes(b"reference")
    training = {
        "status": "completed",
        "completed": {},
        "cpu_seconds": 10,
        "wall_seconds": 12,
        "peak_memory_bytes": 1000,
    }
    model_paths = {"reference": tmp_path / "reference.pt"}
    for replica in range(5):
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
                        "freeze_mode": cfg["freeze_modes"][arm],
                        "kill_reward": cfg["kill_rewards"][arm],
                        "update_every": cfg["update_every"][arm],
                        "l2_strength": cfg["l2_strength"][arm],
                    }
                )
            zip_json(directory / "episodes.json.gz", rows)
            write(
                directory / "result.json",
                {
                    "episodes": 100,
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
    evaluation = {
        "status": "completed",
        "completed": {},
        "cpu_seconds": 10,
        "wall_seconds": 12,
        "peak_memory_bytes": 1000,
    }
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
                            8
                            if artifact.startswith("treatment")
                            else 1
                            if artifact == "reference"
                            else 0
                        )
                    )
                    if suite != "classic-peaceful":
                        kills = 0
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
                            "greedy_actions_sha256": "same-actions",
                            "kill_reward": cfg["kill_rewards"][
                                "treatment" if artifact.startswith("treatment") else "control"
                            ],
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
                    "environment": {"host": "PC"},
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

    import gzip
    from copy import deepcopy

    for suite, field, value, gate in (
        ("coin-heaven", "coins", 4, "opponent_free_invariance"),
        ("classic-peaceful", "kills", 0, "hunting_parent_retention"),
        ("loot-crate", "self_kills", 1, "earlier_task_retention"),
        ("classic-empty", "invalid", 1, "invalid_actions"),
        ("classic-empty", "decision_times_ms", [100], "latency"),
    ):
        backups = []
        for replica in range(1, 6):
            directory = tmp_path / f"evaluation/treatment-r{replica}-{suite}"
            with gzip.open(directory / "episodes.json.gz", "rt") as file:
                original = json.load(file)
            metadata = json.loads((directory / "result.json").read_text())
            backups.append((directory, original, deepcopy(metadata)))
            changed = deepcopy(original)
            for row in changed:
                row["native"]["agents"]["DagobertDuckDQNTask3"][field] = value
            zip_json(directory / "episodes.json.gz", changed)
            metadata["episodes_sha256"] = sha(directory / "episodes.json.gz")
            write(directory / "result.json", metadata)
        negative = analysis.analyze(tmp_path)
        assert not negative["pilot_screen_passed"]
        assert not negative["gates"][gate]
        for directory, original, metadata in backups:
            zip_json(directory / "episodes.json.gz", original)
            write(directory / "result.json", metadata)

    # Round-trip the same portable transfer layout using explicitly synthetic bytes.
    import tarfile

    import scripts.pilot_task3_stability as pilot

    monkeypatch.setattr(pilot, "verify", lambda root: None)
    for name, value in (
        ("binding.json", {"tool_source": "synthetic-test"}),
        ("config.json", cfg),
        ("source-manifest.json", {}),
    ):
        write(tmp_path / name, value)
    (tmp_path / "initial.pt").write_bytes(b"synthetic-initial")
    (tmp_path / "initial-control.pt").write_bytes(b"control")
    (tmp_path / "initial-treatment.pt").write_bytes(b"treatment")
    write(tmp_path / "initialization-verification.json", {})
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

    for replica in range(1, 6):
        directory = tmp_path / f"evaluation/treatment-r{replica}-coin-heaven"
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
    metadata["environment"] = {"host": "PC"}
    write(directory / "result.json", metadata)
    training_directory = tmp_path / "training/control-r1"
    with gzip.open(training_directory / "episodes.json.gz", "rt") as file:
        training_rows = json.load(file)
    training_rows[0]["behavior_epsilon"] = 0.7
    zip_json(training_directory / "episodes.json.gz", training_rows)
    training_metadata = json.loads((training_directory / "result.json").read_text())
    training_metadata["episodes_sha256"] = sha(training_directory / "episodes.json.gz")
    write(training_directory / "result.json", training_metadata)
    with pytest.raises(ValueError, match="exploration schedule"):
        analysis.analyze(tmp_path)


def test_guard_with_real_dqn_loss_clipping_sync_and_reload(monkeypatch):
    import numpy as np

    from agent_code.DagobertDuckDQNTask3 import config as agent_config
    from agent_code.DagobertDuckDQNTask3.model import DQNLearner

    # Main has a 34-input fixture; this test supplies the archived 39-input shape.
    monkeypatch.setattr(agent_config, "FEATURE_COUNT", 39)
    cfg = agent_config.DQNConfig(input_dim=39, target_update_interval=2)
    subject = DQNLearner(config=cfg, seed=173)
    anchor = copy.deepcopy(subject.online_network.state_dict())
    FrozenOpponentInputs(subject, anchor)
    rng = np.random.default_rng(173)
    batch = SimpleNamespace(
        states=rng.normal(size=(64, 39)).astype(np.float32),
        next_states=rng.normal(size=(64, 39)).astype(np.float32),
        action_indices=np.zeros(64, dtype=np.int64),
        rewards=np.ones(64, dtype=np.float32),
        terminals=np.zeros(64, dtype=np.bool_),
        next_action_masks=np.ones((64, 6), dtype=np.bool_),
    )
    assert not subject.train_batch(batch).target_synchronized
    assert subject.train_batch(batch).target_synchronized
    assert all(
        torch.equal(v, subject.target_network.state_dict()[k])
        for k, v in subject.online_network.state_dict().items()
    )
    restored = DQNLearner(config=cfg, seed=173)
    restored.load_state_dict(subject.state_dict())
    FrozenOpponentInputs(restored, anchor)
    subject.train_batch(batch)
    restored.train_batch(batch)
    assert all(
        torch.equal(v, restored.online_network.state_dict()[k])
        for k, v in subject.online_network.state_dict().items()
    )


def test_regularizer_matches_explicit_penalty_and_zero_control():
    from scripts.task3_stability_interventions import RegularizedOpponentInputs

    for coefficient in (0.0, 0.01):
        a, b = learner(), learner()
        anchor = copy.deepcopy(a.online_network.state_dict())
        RegularizedOpponentInputs(a, anchor, coefficient)
        FrozenOpponentInputs(b, anchor)
        torch.manual_seed(45)
        inputs = torch.randn(64, 39)
        a.optimizer.zero_grad(set_to_none=True)
        b.optimizer.zero_grad(set_to_none=True)
        loss_a = a.online_network.layers(inputs).square().mean()
        loss_b = b.online_network.layers(inputs).square().mean()
        weights = dict(b.online_network.named_parameters())["layers.0.weight"]
        loss_b = loss_b + 0.5 * coefficient * weights[:, 26:].square().sum()
        loss_a.backward()
        loss_b.backward()
        ga = dict(a.online_network.named_parameters())["layers.0.weight"].grad
        gb = weights.grad
        torch.testing.assert_close(ga, gb, rtol=1e-6, atol=1e-8)
        assert torch.count_nonzero(ga[:, :26]) == 0
        a.optimizer.step()
        b.optimizer.step()
        for k, v in a.online_network.state_dict().items():
            torch.testing.assert_close(v, b.online_network.state_dict()[k])


def test_update_frequency_keeps_all_transitions_and_rewards():
    from scripts.task3_stability_interventions import update_frequency

    rows, calls = [], []

    class Replay:
        def __len__(self):
            return len(rows)

        def add(self, **kwargs):
            rows.append(kwargs)

        def sample(self, size):
            calls.append(size)
            return None

    subject = SimpleNamespace(
        replay_buffer=Replay(),
        config=SimpleNamespace(replay_warmup=3, batch_size=2),
        learner=SimpleNamespace(
            train_batch=lambda batch: SimpleNamespace(
                loss=1.0, mean_abs_td_error=2.0, target_synchronized=True
            )
        ),
        losses=[],
        absolute_td_errors=[],
        episode_target_synchronizations=0,
        episode_reward=0.0,
    )

    def sentinel(*args, **kwargs):
        return None

    module = SimpleNamespace(_record_transition=sentinel)
    with update_frequency(module, 1):
        assert module._record_transition is sentinel
    with update_frequency(module, 8):
        for index in range(18):
            module._record_transition(
                subject,
                state=None,
                action_index=0,
                reward=1.0,
                next_state=None,
                terminal=index == 17,
            )
    assert module._record_transition is sentinel
    assert len(rows) == 18 and rows[-1]["terminal"]
    assert subject.episode_reward == 18
    assert subject.stability_eligible == 16
    assert calls == [2, 2] and subject.episode_target_synchronizations == 2
    assert len(subject.losses) == 2
