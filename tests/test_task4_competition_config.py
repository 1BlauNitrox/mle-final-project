"""The registered Task 4 comparisons vary one factor over a registered vocabulary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.pilot_task4_competition import (
    PROFILES,
    REGISTERED_OPPONENTS,
    REGISTERED_SCOPES,
    _hashable,
)

ROOT = Path(__file__).resolve().parents[1]
FACTORS = (
    "trainable_scope",
    "training_opponents",
    "potential_scale",
    "update_every",
    "kill_reward",
    "learning_rate",
    "random_episode_period",
)
EXPECTED_FACTOR = {
    "trainable-scope": "trainable_scope",
    "opponent-mixture": "training_opponents",
    "finetune-dose": "learning_rate",
    "lineup-trajectory": "training_opponents",
    "exploration-period": "random_episode_period",
}


def load(profile):
    path = ROOT / f"experiments/2026-09-15-task4-{profile}/config.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_two_registered_profiles_are_the_ones_shipped():
    assert set(PROFILES) == set(EXPECTED_FACTOR)


@pytest.mark.parametrize("profile", PROFILES)
def test_each_comparison_varies_exactly_its_intended_factor(profile):
    cfg = load(profile)
    settings = cfg["arm_settings"]
    varying = {
        key
        for key in FACTORS
        if len({_hashable(settings[arm].get(key)) for arm in cfg["arms"]}) > 1
    }
    assert varying == {EXPECTED_FACTOR[profile]}


@pytest.mark.parametrize("profile", PROFILES)
def test_every_arm_uses_registered_opponents_and_scopes(profile):
    cfg = load(profile)
    for setting in cfg["arm_settings"].values():
        assert setting["trainable_scope"] in REGISTERED_SCOPES
        assert setting["training_opponents"]
        assert all(name in REGISTERED_OPPONENTS for name in setting["training_opponents"])
        # Four agents at most sit on a board, so the agent plus its opponents must fit.
        assert len(setting["training_opponents"]) <= 3


@pytest.mark.parametrize("profile", PROFILES)
def test_the_tournament_suite_is_primary_and_retention_is_measured(profile):
    cfg = load(profile)
    suites = cfg["evaluation_suites"]
    assert cfg["hunting_suite"] == "classic-rule-based"
    assert suites["classic-rule-based"]["opponents"] == ["rule_based_agent"] * 3
    # Opponent-free suites must survive a scope that can move inherited weights.
    assert [] in [s["opponents"] for s in suites.values()]
    assert {"coin-heaven", "loot-crate"} <= set(suites)
    assert "score" in cfg["primary_endpoint"]


@pytest.mark.parametrize("profile", PROFILES)
def test_declared_episode_totals_match_the_seed_tables(profile):
    cfg = load(profile)
    arms, replicas = cfg["arms"], cfg["replicas"]
    assert cfg["training_episodes"] == len(arms) * replicas * cfg["episodes_per_replica_arm"]
    assert len(cfg["training_world_seeds"]) == replicas == len(cfg["replica_agent_seeds"])
    assert all(
        len(seeds) == cfg["episodes_per_replica_arm"] for seeds in cfg["training_world_seeds"]
    )
    per_artifact = cfg["evaluation_repeats"] * sum(
        len(s["world_seeds"]) for s in cfg["evaluation_suites"].values()
    )
    assert cfg["evaluation_episodes"] == per_artifact * (len(arms) * replicas + 1)


def seed_space(cfg):
    values = {s for seeds in cfg["training_world_seeds"] for s in seeds}
    values |= {s for suite in cfg["evaluation_suites"].values() for s in suite["world_seeds"]}
    values |= set(cfg["replica_agent_seeds"]) | set(cfg["smoke"]["world_seeds"])
    values |= {cfg["smoke"]["agent_seed"]}
    return values | {s + 1000000 for s in set(values)}


@pytest.mark.parametrize("profile", PROFILES)
def test_training_evaluation_and_smoke_seeds_stay_separate(profile):
    cfg = load(profile)
    train = {s for seeds in cfg["training_world_seeds"] for s in seeds}
    train |= {s + 1000000 for s in set(train)} | set(cfg["replica_agent_seeds"])
    dev = {s for suite in cfg["evaluation_suites"].values() for s in suite["world_seeds"]}
    dev |= {s + 1000000 for s in set(dev)}
    smoke = set(cfg["smoke"]["world_seeds"]) | {cfg["smoke"]["agent_seed"]}
    assert not train & dev and not train & smoke and not dev & smoke


def test_no_seed_is_shared_with_the_other_task4_profile_or_with_task3():
    spaces = {p: seed_space(load(p)) for p in PROFILES}
    for index, first in enumerate(PROFILES):
        for second in PROFILES[index + 1 :]:
            assert not spaces[first] & spaces[second], f"{first} reuses seeds from {second}"

    task3 = {}
    for name in ("approach-shaping", "update-cadence", "compensated-kill-reward",
                 "opponent-shrinkage"):
        path = ROOT / f"experiments/2026-09-14-task3-{name}/config.json"
        task3[name] = seed_space(json.loads(path.read_text(encoding="utf-8")))
    for profile, space in spaces.items():
        for name, other in task3.items():
            assert not space & other, f"{profile} reuses seeds from task3 {name}"
