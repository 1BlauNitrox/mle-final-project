"""Joint-gradient, provenance, replay recovery and confirmation contracts."""

import json
from copy import deepcopy
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from agent_code.DagobertDuckDQNAntiLoop.config import DQNConfig
from agent_code.DagobertDuckDQNAntiLoop.model import DQNLearner
from agent_code.DagobertDuckDQNAntiLoop.replay import ReplayBuffer
from scripts.confirm_teacher_loop import confirmation
from scripts.curriculum_io import read, write
from scripts.run_teacher_loop import CONFIG, prepared_usage, sources, training_deadline_breach
from training.teacher_loop import TeacherTracker, distillation_loss, exposure_gate


@pytest.mark.parametrize("inherited_cpu", [None, 1060.4375, 800.0])
def test_preparation_retains_prior_cpu_and_original_start(tmp_path, monkeypatch, inherited_cpu):
    from scripts import run_teacher_loop as runner

    monkeypatch.setattr(runner.time, "time", lambda: 2000.0)
    monkeypatch.setattr(
        runner.psutil, "Process", lambda: SimpleNamespace(cpu_times=lambda: (12.0, 3.0))
    )
    cfg = {"prior_usage": {"pc": {"cpu_seconds": 900.0}}}
    if inherited_cpu is not None:
        write(
            tmp_path / "inherited-resources.json",
            {
                "cpu_seconds": inherited_cpu,
                "first_start": 1000.0,
            },
        )
    usage = prepared_usage(tmp_path, cfg, "pc", 1900.0)
    assert usage["cpu_seconds"] == max(900.0, inherited_cpu or 0) + 15.0
    assert usage["first_start"] == (1000.0 if inherited_cpu is not None else 1900.0)
    assert usage["wall_seconds"] == 2000.0 - usage["first_start"]
    write(tmp_path / "resources.json", usage)
    assert runner.initial_usage(tmp_path, cfg, "pc") == usage


def world(step, position=(3, 3)):
    field = np.zeros((17, 17), dtype=int)
    field[[0, -1], :] = -1
    field[:, [0, -1]] = -1
    return {
        "round": 1,
        "step": step,
        "field": field,
        "self": ("test", 0, True, position),
        "others": [],
        "coins": [],
        "bombs": [],
        "explosion_map": np.zeros_like(field),
    }


def policy(seed=1):
    config = DQNConfig(batch_size=2, replay_capacity=8, replay_warmup=2)
    return SimpleNamespace(
        config=config,
        learner=DQNLearner(config=config, seed=seed),
        replay_buffer=ReplayBuffer(capacity=8, seed=seed),
        pending_transition=None,
        episode_reward=0,
        losses=[],
        absolute_td_errors=[],
        episode_target_synchronizations=0,
    )


def tracker(agent, *, state=None, warmup=2, weight=1):
    return TeacherTracker(
        state or {"transitions": 0, "generated": {}, "sampled": {}, "origins": []},
        capacity=8,
        every=1,
        maximum_updates=100,
        origin="classic",
        warmup=warmup,
        weight=weight,
        teacher=deepcopy(agent.learner.online_network),
    )


def record(t, agent, step):
    t.observe(world(step))
    t.record(
        agent,
        state=np.full(56, step / 100, dtype=np.float32),
        action_index=0,
        reward=1.0,
        next_state=None,
        terminal=True,
    )


def test_loss_excludes_loop_rows_illegal_actions_and_teacher_gradients():
    student = torch.randn(3, 6, requires_grad=True)
    teacher = torch.randn(3, 6, requires_grad=True)
    legal = torch.ones(3, 6, dtype=torch.bool)
    legal[:, 5] = False
    loss = distillation_loss(student, teacher, legal, torch.tensor([True, False, True]))
    loss.backward()
    assert teacher.grad is None
    assert torch.count_nonzero(student.grad[1]) == 0
    assert torch.count_nonzero(student.grad[:, 5]) == 0
    assert torch.count_nonzero(student.grad[0, :5]) > 0


def test_no_retention_rows_have_zero_loss_and_gradient():
    student = torch.randn(2, 6, requires_grad=True)
    loss = distillation_loss(
        student,
        student.detach() + 1,
        torch.ones(2, 6, dtype=torch.bool),
        torch.zeros(2, dtype=torch.bool),
    )
    loss.backward()
    assert loss.item() == 0 and torch.count_nonzero(student.grad) == 0


def test_joint_loss_is_applied_before_gradient_clipping():
    a, b = policy(), policy()
    for i in (1, 2):
        a.replay_buffer.add(
            state=np.full(56, i, dtype=np.float32),
            action_index=0,
            reward=1.0,
            next_state=None,
            terminal=True,
        )
    batch = a.replay_buffer.sample(2)
    a.learner.auxiliary_loss = lambda states, values: 2 * values.square().mean()
    a.learner.train_batch(batch)
    values = b.learner.online_network(torch.from_numpy(batch.states))
    expected = F.smooth_l1_loss(values[:, 0], torch.from_numpy(batch.rewards))
    expected = expected + 2 * values.square().mean()
    b.learner.optimizer.zero_grad(set_to_none=True)
    expected.backward()
    torch.nn.utils.clip_grad_norm_(
        b.learner.online_network.parameters(), b.config.gradient_clip_norm
    )
    b.learner.optimizer.step()
    for left, right in zip(
        a.learner.online_network.parameters(), b.learner.online_network.parameters(), strict=True
    ):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_warmup_suppresses_updates_and_zero_weight_matches_original_update():
    a = policy()
    t = tracker(a, weight=0)
    record(t, a, 1)
    record(t, a, 2)
    assert a.learner.update_steps == 0 and not t.state["sampled"]
    b = policy()
    b.replay_buffer.load_state_dict(a.replay_buffer.state_dict())
    b.replay_buffer.add(
        state=np.full(56, 0.03, dtype=np.float32),
        action_index=0,
        reward=1.0,
        next_state=None,
        terminal=True,
    )
    b.learner.train_batch(b.replay_buffer.sample(2))
    record(t, a, 3)
    assert a.learner.update_steps == 1
    assert t.state["teacher_loss_updates"] == 0
    for left, right in zip(
        a.learner.online_network.parameters(), b.learner.online_network.parameters(), strict=True
    ):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_delayed_pending_metadata_uses_its_own_observation():
    a = policy()
    t = tracker(a)
    first = world(1)
    first["field"][3, 2] = 1
    t.observe(first)
    t.observe(world(2))
    a.pending_transition = SimpleNamespace(identity=(1, 1), state=np.zeros(56, dtype=np.float32))
    t.record(
        a,
        state=a.pending_transition.state,
        action_index=1,
        reward=0.0,
        next_state=None,
        terminal=True,
    )
    assert not t.metadata[-1]["legal"][0]
    assert (1, 2) in t.observations and (1, 1) not in t.observations


def test_teacher_stays_fixed_and_resume_reproduces_updates():
    a = policy()
    t = tracker(a)
    original_teacher = deepcopy(t.teacher.state_dict())
    for step in range(1, 6):
        record(t, a, step)
    b = policy()
    b.learner.load_state_dict(a.learner.state_dict())
    b.replay_buffer.load_state_dict(a.replay_buffer.state_dict())
    resumed = tracker(b, state=json.loads(json.dumps(t.snapshot())))
    resumed.teacher.load_state_dict(original_teacher)
    for step in range(6, 9):
        # Recovery occurs at a game boundary; use identical new-round histories.
        record(t, a, step)
        record(resumed, b, step)
    assert t.snapshot() == resumed.snapshot()
    for key, value in original_teacher.items():
        torch.testing.assert_close(t.teacher.state_dict()[key], value, rtol=0, atol=0)
    for left, right in zip(
        a.learner.online_network.parameters(), b.learner.online_network.parameters(), strict=True
    ):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_causal_loop_classification_and_unknown_history_exclusion():
    a = policy()
    t = tracker(a)
    for step in range(1, 25):
        t.observe(world(step, (3, 3 + step % 2)))
    assert t.observations[(1, 24)]["loop"]
    hazard = world(25)
    hazard["bombs"] = [((8, 8), 3)]
    t.observe(hazard)
    assert not t.observations[(1, 25)]["loop"]
    assert not all(
        exposure_gate(
            t.state, {"generated_loop": 1, "sampled_loop": 1, "sampled_retention": 1}
        ).values()
    )


def test_training_deadline_allows_evaluation_but_never_training(tmp_path):
    cfg = read(CONFIG)
    deadline = datetime.fromisoformat(cfg["limits"]["training_stop_utc"]).timestamp()
    write(tmp_path / "pairs/r1/work-stage.json", {"stage": "training"})
    assert training_deadline_breach(tmp_path, cfg, deadline - 1) is None
    assert training_deadline_breach(tmp_path, cfg, deadline)
    write(tmp_path / "pairs/r1/work-stage.json", {"stage": "evaluation"})
    assert training_deadline_breach(tmp_path, cfg, deadline) is None
    assert "training/teacher_loop.py" in sources()


def test_confirmation_rejects_regression_invalids_and_wrong_seeds(tmp_path):
    cfg = read(CONFIG)
    cfg["bootstrap_resamples"] = 20
    for setting in cfg["evaluation"]["confirmation"].values():
        setting["seeds"][1] = setting["seeds"][0] + 1
    for name in ("candidate", "reference"):
        for suite, setting in cfg["evaluation"]["confirmation"].items():
            observations = [
                {
                    "world_seed": seed,
                    "slot": i % (len(setting["opponents"]) + 1),
                    "opponents": setting["opponents"],
                    "scenario": setting["scenario"],
                    "epsilon": 0,
                    "native": {
                        "score": 3,
                        "kills": 1,
                        "survived": 1,
                        "self_kills": 0,
                        "coins": 3,
                        "initially_available_coins": 9,
                        "invalid": 0,
                        "decision_times_ms": [1.0, 2.0],
                    },
                    "loop_windows": {
                        "eligible": 100,
                        "looping": 50 if name == "candidate" else 100,
                    },
                }
                for i, seed in enumerate(range(*[setting["seeds"][0], setting["seeds"][1] + 1]))
            ]
            write(tmp_path / name / f"{suite}.json", {"rows": observations})
    candidate, reference = tmp_path / "candidate", tmp_path / "reference"
    assert confirmation(candidate, reference, cfg)["passed"]
    path = candidate / "classic.json"
    original = read(path)
    for metric in ("invalid", "kills", "score"):
        changed = deepcopy(original)
        for row in changed["rows"]:
            row["native"][metric] += 1 if metric == "invalid" else -1
        write(path, changed)
        assert not confirmation(candidate, reference, cfg)["passed"]
    original["rows"][0]["world_seed"] += 50
    write(path, original)
    with pytest.raises(ValueError, match="seeds/count"):
        confirmation(candidate, reference, cfg)
