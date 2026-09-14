"""The registered approach shaping and update interval are training-only and exact."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from agent_code.DagobertDuckDQNTask3 import train
from agent_code.DagobertDuckDQNTask3.features import normalize_features, state_to_features
from scripts.task3_approach_interventions import (
    approach_potential,
    nearest_opponent_distance,
    potential_reward,
    training_intervention,
)
from tests.test_DagobertDuckDQNTask3_opponents import make_field, make_state
from tests.test_DagobertDuckDQNTask3_successor import make_agent

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ("approach-shaping", "update-cadence")
GAMMA = 0.9


@pytest.fixture(autouse=True)
def isolated_checkpoint(tmp_path, monkeypatch):
    """`end_of_round` saves a checkpoint, so never let a test reach the fixture."""
    committed = Path(train.CHECKPOINT_PATH)
    before = committed.read_bytes()
    monkeypatch.setattr(train, "CHECKPOINT_PATH", tmp_path / "checkpoint.pt")
    yield
    assert committed.read_bytes() == before


def potential(state, scale):
    return approach_potential(state, scale=scale, discount_factor=GAMMA)


def wide_state(distance, *, position=(1, 1), step=1):
    """Return a large empty board with one opponent at a chosen Manhattan distance."""
    field = make_field(size=40)
    others = None
    if distance is not None:
        others = [("target", 0, True, (position[0] + distance, position[1]))]
    state = make_state(position=position, others=others, field=field)
    state["step"] = step
    return state


def features_for(state):
    return normalize_features(state_to_features(state))


# --- potential function -------------------------------------------------------


def test_distance_uses_the_nearest_public_opponent():
    assert nearest_opponent_distance(None) is None
    assert nearest_opponent_distance(make_state()) is None
    state = make_state(
        position=(3, 3),
        others=[("far", 0, True, (5, 5)), ("near", 0, True, (3, 2))],
    )
    assert nearest_opponent_distance(state) == 1


def test_distance_is_a_plain_int_so_evidence_rows_stay_serializable():
    """Framework positions can be NumPy integers, which json.dumps refuses."""
    state = make_state(
        position=(np.int64(3), np.int64(3)),
        others=[("target", 0, True, (np.int64(3), np.int64(6)))],
    )
    distance = nearest_opponent_distance(state)
    assert type(distance) is int
    assert json.dumps({"opponent_distance_sum": distance}) == '{"opponent_distance_sum": 3}'


def test_potential_is_zero_without_an_opponent_and_decays_with_distance():
    assert potential(None, 3.0) == 0.0
    assert potential(make_state(), 3.0) == 0.0
    values = [potential(wide_state(d), 1.0) for d in range(1, 29)]
    assert values == sorted(values, reverse=True)
    assert values[0] == pytest.approx(GAMMA)
    assert values[-1] == pytest.approx(GAMMA**28)
    assert potential(wide_state(4), 3.0) == pytest.approx(3.0 * potential(wide_state(4), 1.0))


def test_shaping_is_exactly_neutral_for_closing_the_distance_by_one():
    """The registered potential cancels discounting along an approach."""
    for distance in (2, 8, 16, 24):
        here = potential(wide_state(distance), 3.0)
        closer = potential(wide_state(distance - 1), 3.0)
        further = potential(wide_state(distance + 1), 3.0)
        approach = potential_reward(here, closer, terminal=False, discount_factor=GAMMA)
        stay = potential_reward(here, here, terminal=False, discount_factor=GAMMA)
        retreat = potential_reward(here, further, terminal=False, discount_factor=GAMMA)
        assert approach == pytest.approx(0.0, abs=1e-12)
        assert stay == pytest.approx(-(1 - GAMMA) * here)
        assert retreat == pytest.approx(-(1 - GAMMA**2) * here)
        assert approach > stay > retreat


def test_disabled_scale_makes_the_potential_identically_zero():
    assert all(potential(wide_state(d), 0.0) == 0.0 for d in (1, 7, 20))


def test_unregistered_scale_or_discount_is_refused():
    with pytest.raises(ValueError, match="Unregistered approach-potential scale"):
        potential(make_state(), 0.25)
    with pytest.raises(ValueError, match="Discount factor"):
        approach_potential(make_state(), scale=1.0, discount_factor=0.0)


# --- shaping arithmetic -------------------------------------------------------


def test_potential_reward_matches_the_registered_formula():
    assert potential_reward(0.4, 0.5, terminal=False, discount_factor=GAMMA) == pytest.approx(
        GAMMA * 0.5 - 0.4
    )
    assert potential_reward(0.4, None, terminal=True, discount_factor=GAMMA) == pytest.approx(-0.4)
    with pytest.raises(ValueError, match="Terminal shaping"):
        potential_reward(0.4, 0.5, terminal=True, discount_factor=GAMMA)
    with pytest.raises(ValueError, match="require a next potential"):
        potential_reward(0.4, None, terminal=False, discount_factor=GAMMA)
    with pytest.raises(ValueError, match="Discount factor"):
        potential_reward(0.4, 0.5, terminal=False, discount_factor=1.5)


def test_shaping_telescopes_to_minus_the_initial_potential():
    """A trajectory's discounted shaping is exactly -Phi(s0), so it adds no return."""
    distances = [12, 11, 10, 11, 10, 9, 8]
    potentials = [potential(wide_state(d), 3.0) for d in distances]
    total = 0.0
    for step in range(len(potentials) - 1):
        total += GAMMA**step * potential_reward(
            potentials[step], potentials[step + 1], terminal=False, discount_factor=GAMMA
        )
    total += GAMMA ** (len(potentials) - 1) * potential_reward(
        potentials[-1], None, terminal=True, discount_factor=GAMMA
    )
    assert total == pytest.approx(-potentials[0])


# --- training integration -----------------------------------------------------


def drive(agent, states, *, terminal_state):
    """Replay one scripted episode through the framework training callbacks."""
    for index in range(len(states) - 1):
        train.game_events_occurred(agent, states[index], "RIGHT", states[index + 1], [])
    train.end_of_round(agent, terminal_state, "RIGHT", ["SURVIVED_ROUND"])


def scripted_states(count=6):
    return [wide_state(12 - index, step=index + 1) for index in range(count)]


def test_neutral_intervention_reproduces_the_unpatched_runtime_exactly():
    import torch

    states = scripted_states()
    plain = make_agent()
    train.setup_training(plain)
    drive(plain, states, terminal_state=states[-2])

    patched = make_agent()
    train.setup_training(patched)
    with training_intervention(
        train, potential_scale=0.0, update_every=1, discount_factor=GAMMA
    ):
        drive(patched, states, terminal_state=states[-2])

    expected = plain.replay_buffer.state_dict()
    actual = patched.replay_buffer.state_dict()
    assert set(expected) == set(actual)
    for key in ("states", "action_indices", "rewards", "next_states", "terminals"):
        assert np.array_equal(expected[key], actual[key]), key
    assert plain.learner.update_steps == patched.learner.update_steps > 0
    assert plain.episode_reward == pytest.approx(patched.episode_reward)
    for name, tensor in plain.learner.online_network.state_dict().items():
        assert torch.equal(tensor, patched.learner.online_network.state_dict()[name]), name


def test_intervention_restores_every_patched_callback():
    originals = (train.game_events_occurred, train.end_of_round, train._record_transition)
    with training_intervention(
        train, potential_scale=3.0, update_every=8, discount_factor=GAMMA
    ):
        assert train._record_transition is not originals[2]
    assert (train.game_events_occurred, train.end_of_round, train._record_transition) == originals


@pytest.mark.parametrize(
    "scale,interval,message",
    [
        (0.5, 1, "Unregistered approach-potential scale"),
        (0.0, 3, "Unregistered update interval"),
    ],
)
def test_unregistered_intervention_values_are_refused(scale, interval, message):
    with (
        pytest.raises(ValueError, match=message),
        training_intervention(
            train, potential_scale=scale, update_every=interval, discount_factor=GAMMA
        ),
    ):
        pass


@pytest.mark.parametrize(
    "distances,sign",
    [((12, 11, 10), 0.0), ((12, 12, 12), -1.0), ((12, 13, 14), -1.0)],
    ids=["approach", "stay", "retreat"],
)
def test_shaping_adds_the_registered_term_to_stored_rewards(distances, sign):
    states = [wide_state(d, step=index + 1) for index, d in enumerate(distances)]
    agent = make_agent()
    train.setup_training(agent)
    with training_intervention(
        train, potential_scale=3.0, update_every=1, discount_factor=GAMMA
    ):
        train.game_events_occurred(agent, states[0], "RIGHT", states[1], [])
        assert len(agent.replay_buffer) == 0  # the first transition is still pending
        train.game_events_occurred(agent, states[1], "RIGHT", states[2], [])

    stored = agent.replay_buffer._transitions[-1]
    expected = potential_reward(
        potential(states[0], 3.0),
        potential(states[1], 3.0),
        terminal=False,
        discount_factor=GAMMA,
    )
    assert stored.terminal is False
    assert stored.reward == pytest.approx(train.reward_from_events([]) + expected)
    if sign:
        assert expected < 0
    else:
        assert expected == pytest.approx(0.0, abs=1e-12)


def test_terminal_shaping_uses_minus_the_last_state_potential():
    states = scripted_states(4)
    agent = make_agent()
    train.setup_training(agent)
    with training_intervention(
        train, potential_scale=3.0, update_every=1, discount_factor=GAMMA
    ):
        train.game_events_occurred(agent, states[0], "RIGHT", states[1], [])
        train.end_of_round(agent, states[0], "RIGHT", ["SURVIVED_ROUND"])

    stored = agent.replay_buffer._transitions[-1]
    assert stored.terminal is True
    assert stored.reward == pytest.approx(
        train.reward_from_events(["SURVIVED_ROUND"]) - potential(states[0], 3.0)
    )


def test_opponent_free_states_receive_no_shaping():
    empty = [make_state(position=(3, 3)) for _ in range(3)]
    agent = make_agent()
    train.setup_training(agent)
    with training_intervention(
        train, potential_scale=3.0, update_every=1, discount_factor=GAMMA
    ):
        drive(agent, empty, terminal_state=empty[0])
    rewards = agent.replay_buffer.state_dict()["rewards"]
    assert np.allclose(rewards[:-1], train.reward_from_events([]))


def test_update_interval_throttles_updates_but_keeps_every_transition():
    state = features_for(wide_state(5))
    for interval, expected in ((1, 16), (8, 2)):
        agent = make_agent()
        train.setup_training(agent)
        with training_intervention(
            train, potential_scale=0.0, update_every=interval, discount_factor=GAMMA
        ):
            for _ in range(16):
                train._record_transition(
                    agent,
                    state=state,
                    action_index=0,
                    reward=1.0,
                    next_state=state,
                    terminal=False,
                )
        assert agent.learner.update_steps == expected
        assert agent.episode_reward == pytest.approx(16.0)
        assert len(agent.replay_buffer) == min(16, agent.config.replay_capacity)


# --- registered configurations ------------------------------------------------


def load(profile):
    path = ROOT / f"experiments/2026-09-14-task3-{profile}/config.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("profile", PROFILES)
def test_registered_configuration_is_internally_consistent(profile):
    cfg = load(profile)
    arms = cfg["arms"]
    assert cfg["profile"] == profile
    assert list(cfg["arm_settings"]) == arms and "control" in arms
    assert cfg["training_episodes"] == len(arms) * cfg["replicas"] * cfg["episodes_per_replica_arm"]
    per_artifact = cfg["evaluation_repeats"] * sum(
        len(s["world_seeds"]) for s in cfg["evaluation_suites"].values()
    )
    assert cfg["evaluation_episodes"] == per_artifact * (len(arms) * cfg["replicas"] + 1)
    assert all(
        len(seeds) == cfg["episodes_per_replica_arm"] for seeds in cfg["training_world_seeds"]
    )
    assert len(cfg["training_world_seeds"]) == cfg["replicas"] == len(cfg["replica_agent_seeds"])
    assert cfg["hunting_suite"] in cfg["evaluation_suites"]
    assert cfg["evaluation_suites"][cfg["hunting_suite"]]["opponents"] == ["peaceful_agent"]
    assert cfg["smoke"]["arm"] in arms


@pytest.mark.parametrize("profile", PROFILES)
def test_exactly_one_factor_varies_between_arms(profile):
    cfg = load(profile)
    settings = cfg["arm_settings"]
    varying = {
        key
        for key in ("potential_scale", "update_every", "kill_reward", "learning_rate")
        if len({settings[arm][key] for arm in cfg["arms"]}) > 1
    }
    assert len(varying) == 1
    assert varying == ({"potential_scale"} if profile == "approach-shaping" else {"update_every"})
    assert all(setting["kill_reward"] == 5.0 for setting in settings.values())


@pytest.mark.parametrize("profile", PROFILES)
def test_registered_seeds_are_internally_disjoint(profile):
    cfg = load(profile)
    train_seeds = {s for seeds in cfg["training_world_seeds"] for s in seeds}
    train_seeds |= {s + 1000000 for s in set(train_seeds)} | set(cfg["replica_agent_seeds"])
    dev = {s for suite in cfg["evaluation_suites"].values() for s in suite["world_seeds"]}
    dev |= {s + 1000000 for s in set(dev)}
    smoke = set(cfg["smoke"]["world_seeds"]) | {cfg["smoke"]["agent_seed"]}
    assert not train_seeds & dev
    assert not train_seeds & smoke
    assert not dev & smoke


def test_the_two_profiles_do_not_share_any_seed():
    def everything(cfg):
        values = {s for seeds in cfg["training_world_seeds"] for s in seeds}
        values |= {s for suite in cfg["evaluation_suites"].values() for s in suite["world_seeds"]}
        values |= set(cfg["replica_agent_seeds"]) | set(cfg["smoke"]["world_seeds"])
        values |= {cfg["smoke"]["agent_seed"]}
        return values | {s + 1000000 for s in set(values)}

    first, second = (everything(load(profile)) for profile in PROFILES)
    assert not first & second
