import copy
import json

import pytest

from scripts.probe_task3_exploration import CONFIG, decide, observe, validate


def jobs():
    return [
        {
            "world_seed": seed,
            "epsilon": epsilon,
            "status": "completed",
            "weights_unchanged": True,
            "survival_steps": 10 if epsilon == 1 else 30,
            "coins": 0,
            "attack_steps": 1 if seed < 1631103 else 0,
        }
        for seed in range(1631101, 1631121)
        for epsilon in (1.0, 0.2, 0.0)
    ]


def test_readiness_is_not_checkpoint_selection_or_compute_authorization():
    result = decide(jobs())
    assert result["phase_b_ready"]
    assert result["selected_checkpoint"] is None
    assert not result["long_training_authorized"]


def test_no_greedy_fallback_when_low_epsilon_fails():
    data = jobs()
    for row in data:
        if row["epsilon"] == 0.2:
            row["survival_steps"] = 9
    assert not decide(data)["phase_b_ready"]


@pytest.mark.parametrize("change", ["missing", "duplicate", "failed", "mutable"])
def test_rejects_incomplete_or_invalid_matrix(change):
    data = jobs()
    if change == "missing":
        data.pop()
    elif change == "duplicate":
        data[-1] = dict(data[0])
    elif change == "failed":
        data[0]["status"] = "failed"
    else:
        data[0]["weights_unchanged"] = False
    with pytest.raises(ValueError):
        decide(data)


def test_exposure_and_coin_filters_are_conjunctive():
    data = jobs()
    for row in data:
        if row["epsilon"] == 0.2:
            row["attack_steps"] = 0
    assert not decide(data)["gates"]["attack_exposure"]
    for row in data:
        if row["epsilon"] == 1.0:
            row["coins"] = 1
    assert not decide(data)["gates"]["coins"]


def test_opportunity_requires_confirmed_placement_and_legal_availability():
    features = [0] * 39
    features[30] = features[31] = 1
    state = {
        "step": 1,
        "self": ["agent", 0, True, [1, 1]],
        "others": [["opponent", 0, True, [1, 3]]],
    }
    row = observe(features, state, ["BOMB_DROPPED"], "BOMB", [True] * 6)
    assert row["opponent_distance"] == 2
    assert row["safe_cratefree_placement"] and row["available_safe_attack"]
    assert row["wasteful_safe_attack_penalty"] == -0.5
    row = observe(features, state, [], "BOMB", [True] * 5 + [False])
    assert not row["safe_cratefree_placement"]
    assert not row["available_safe_attack"]
    features[19] = 1
    assert not observe(features, state, ["BOMB_DROPPED"], "BOMB", [True] * 6)[
        "safe_cratefree_placement"
    ]


def test_registered_configuration_rejects_matrix_budget_and_gate_changes():
    config = json.loads(CONFIG.read_text())
    validate(config)
    for mutate in (
        lambda c: c["epsilons"].append(0.1),
        lambda c: c["limits"].update(wall_seconds=901),
        lambda c: c["readiness_filter"].update(strictly_longer_pairs_min=1),
    ):
        modified = copy.deepcopy(config)
        mutate(modified)
        with pytest.raises(ValueError):
            validate(modified)


def test_resource_sampling_counts_descendants_and_retains_exited_cpu():
    from types import SimpleNamespace
    from unittest.mock import Mock

    from scripts.probe_task3_exploration import sample_tree

    parent, child = Mock(pid=10), Mock(pid=11)
    parent.children.return_value = [child]
    parent.create_time.return_value = child.create_time.return_value = 100
    parent.cpu_times.return_value = SimpleNamespace(user=2, system=1)
    child.cpu_times.return_value = SimpleNamespace(user=4, system=1)
    parent.memory_info.return_value = SimpleNamespace(rss=100)
    child.memory_info.return_value = SimpleNamespace(rss=200)
    ledger = {}
    assert sample_tree(parent, ledger) == (8, 300)
    parent.children.return_value = []
    assert sample_tree(parent, ledger) == (8, 100)


def test_stop_owned_targets_only_owned_child_tree(monkeypatch):
    from unittest.mock import Mock

    import psutil

    from scripts.probe_task3_exploration import stop_owned

    child, parent, descendant = Mock(pid=42), Mock(), Mock()
    parent.children.return_value = [descendant]
    lookup = Mock(return_value=parent)
    monkeypatch.setattr(psutil, "Process", lookup)
    stop_owned(child)
    lookup.assert_called_once_with(42)
    parent.children.assert_called_once_with(recursive=True)
    descendant.kill.assert_called_once_with()
    parent.kill.assert_called_once_with()
    child.wait.assert_called_once_with()
