"""The follow-up interventions gate replay and shrink opponent weights as registered."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from agent_code.DagobertDuckDQNTask3 import train
from scripts.task3_approach_interventions import (
    FIRST_LAYER,
    OPPONENT_PREFIX,
    RegularizedOpponentInputs,
    training_intervention,
)
from tests.test_DagobertDuckDQNTask3_successor import make_agent
from tests.test_task3_approach_interventions import wide_state

ROOT = Path(__file__).resolve().parents[1]
GAMMA = 0.9
GATE = 8


@pytest.fixture(autouse=True)
def isolated_checkpoint(tmp_path, monkeypatch):
    committed = Path(train.CHECKPOINT_PATH)
    before = committed.read_bytes()
    monkeypatch.setattr(train, "CHECKPOINT_PATH", tmp_path / "checkpoint.pt")
    yield
    assert committed.read_bytes() == before


# Alternating near/far, starting near so both arms clear replay warm-up together.
DISTANCES = [3, 3, 20, 3, 20, 20, 3, 4, 25, 3, 2, 18]


def drive(gate):
    states = [wide_state(d, step=i + 1) for i, d in enumerate(DISTANCES)]
    agent = make_agent()
    train.setup_training(agent)
    with training_intervention(
        train,
        potential_scale=1.0,
        update_every=1,
        discount_factor=GAMMA,
        proximity_gate=gate,
    ):
        for index in range(len(states) - 1):
            train.game_events_occurred(agent, states[index], "RIGHT", states[index + 1], [])
        train.end_of_round(agent, states[-2], "RIGHT", ["SURVIVED_ROUND"])
    return agent


def test_the_gate_withholds_far_transitions_from_replay():
    ungated, gated = drive(None), drive(GATE)

    # Eleven transitions are recorded from twelve states: ten finalized
    # non-terminal plus the pending one, which end_of_round turns terminal
    # rather than adding a twelfth. Their start states are DISTANCES[:-1].
    starts = DISTANCES[:-1]
    admitted = sum(d <= GATE for d in starts)
    assert 0 < admitted < len(starts)

    assert len(ungated.replay_buffer) == min(len(starts), ungated.config.replay_capacity)
    assert len(gated.replay_buffer) == min(admitted, gated.config.replay_capacity)
    assert getattr(gated, "approach_gated_out", 0) == len(starts) - admitted
    assert getattr(ungated, "approach_gated_out", 0) == 0


def test_gating_does_not_change_how_many_optimizer_steps_run():
    """The registered claim: the gate changes the experience, not the gradient count."""
    ungated, gated = drive(None), drive(GATE)
    assert ungated.learner.update_steps == gated.learner.update_steps > 0


def test_gating_leaves_the_shaped_reward_accounting_untouched():
    ungated, gated = drive(None), drive(GATE)
    assert ungated.episode_reward == pytest.approx(gated.episode_reward)


def test_an_unregistered_gate_is_refused():
    with (
        pytest.raises(ValueError, match="Unregistered proximity gate"),
        training_intervention(
            train, potential_scale=0.0, update_every=1, discount_factor=GAMMA, proximity_gate=5
        ),
    ):
        pass


# --- opponent-weight shrinkage ----------------------------------------------


def guarded(coefficient):
    agent = make_agent()
    anchor = deepcopy(agent.learner.online_network.state_dict())
    return agent, RegularizedOpponentInputs(agent.learner, anchor, coefficient)


def test_shrinkage_adds_lambda_times_weight_on_the_opponent_columns_only():
    agent, guard = guarded(1.0)
    weights = agent.learner.online_network.state_dict()[FIRST_LAYER]
    result = guard.mask_gradient(torch.zeros_like(weights))

    assert torch.equal(result[:, :OPPONENT_PREFIX], torch.zeros_like(result[:, :OPPONENT_PREFIX]))
    assert torch.allclose(result[:, OPPONENT_PREFIX:], weights[:, OPPONENT_PREFIX:])


def test_the_coefficient_scales_the_penalty_and_zero_disables_it():
    agent, guard = guarded(0.1)
    weights = agent.learner.online_network.state_dict()[FIRST_LAYER]
    tenth = guard.mask_gradient(torch.zeros_like(weights))
    assert torch.allclose(tenth[:, OPPONENT_PREFIX:], 0.1 * weights[:, OPPONENT_PREFIX:])

    _, plain = guarded(0.0)
    zeros = torch.zeros_like(weights)
    assert torch.equal(plain.mask_gradient(zeros), zeros)


def test_shrinkage_still_refuses_to_move_inherited_weights():
    agent, guard = guarded(1.0)
    guard.validate()
    with pytest.raises(ValueError, match="Frozen online parameter moved"):
        with torch.no_grad():
            agent.learner.online_network.state_dict()[FIRST_LAYER][:, 0] += 1.0
        guard.validate()


def test_an_unregistered_coefficient_is_refused():
    with pytest.raises(ValueError, match="Unregistered opponent L2 coefficient"):
        guarded(0.5)


# --- registered configurations ----------------------------------------------

FOLLOW_UPS = ("proximity-gating", "opponent-shrinkage")
EXPECTED_FACTOR = {"proximity-gating": "proximity_gate", "opponent-shrinkage": "l2_opponent"}


def load(profile):
    path = ROOT / f"experiments/2026-09-14-task3-{profile}/config.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("profile", FOLLOW_UPS)
def test_follow_up_varies_exactly_its_intended_factor(profile):
    cfg = load(profile)
    settings = cfg["arm_settings"]
    varying = {
        key
        for key in ("potential_scale", "update_every", "kill_reward", "learning_rate",
                    "proximity_gate", "l2_opponent")
        if len({settings[arm][key] for arm in cfg["arms"]}) > 1
    }
    assert varying == {EXPECTED_FACTOR[profile]}
    # Every arm sits on the best measured baseline from the first pair.
    assert all(s["potential_scale"] == 1.0 and s["update_every"] == 8 for s in settings.values())
    assert all(s["kill_reward"] == 5.0 for s in settings.values())


@pytest.mark.parametrize("profile", FOLLOW_UPS)
def test_follow_up_buys_replicas_rather_than_worlds(profile):
    cfg = load(profile)
    assert cfg["replicas"] >= 8
    assert len(cfg["evaluation_suites"]["classic-peaceful"]["world_seeds"]) == 40
    assert cfg["decision_thresholds"]["nonworse_replicas"] == cfg["replicas"] - 1
    assert cfg["issue"] is None and cfg["issue_reference"]


@pytest.mark.parametrize("profile", FOLLOW_UPS)
def test_follow_up_seed_blocks_cannot_overlap_training_and_evaluation(profile):
    cfg = load(profile)
    training = {s for seeds in cfg["training_world_seeds"] for s in seeds}
    training |= {s + 1000000 for s in set(training)} | set(cfg["replica_agent_seeds"])
    dev = {s for suite in cfg["evaluation_suites"].values() for s in suite["world_seeds"]}
    dev |= {s + 1000000 for s in set(dev)}
    smoke = set(cfg["smoke"]["world_seeds"]) | {cfg["smoke"]["agent_seed"]}
    assert not training & dev and not training & smoke and not dev & smoke
    assert len(training) == cfg["replicas"] * cfg["episodes_per_replica_arm"] * 2 + cfg["replicas"]


def test_no_seed_is_shared_across_any_of_the_four_profiles():
    def everything(cfg):
        values = {s for seeds in cfg["training_world_seeds"] for s in seeds}
        values |= {s for suite in cfg["evaluation_suites"].values() for s in suite["world_seeds"]}
        values |= set(cfg["replica_agent_seeds"]) | set(cfg["smoke"]["world_seeds"])
        values |= {cfg["smoke"]["agent_seed"]}
        return values | {s + 1000000 for s in set(values)}

    profiles = ["approach-shaping", "update-cadence", *FOLLOW_UPS]
    spaces = {p: everything(load(p)) for p in profiles}
    for index, first in enumerate(profiles):
        for second in profiles[index + 1 :]:
            assert not spaces[first] & spaces[second], f"{first} and {second} share seeds"
