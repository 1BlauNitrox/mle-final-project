"""Tests for issue #103's reward-variant selection.

`config.REWARDS` itself is resolved once at import time from whichever
process environment variable was set before the module first loaded (see
`training/run_plan.py`'s `BOMBERMAN_DQN_REWARD_VARIANT` environment
override), so these tests exercise the resolution function directly with
monkeypatched environment variables rather than relying on import order.
"""

from __future__ import annotations

import pytest

from agent_code.DagobertDuckDQNTask2 import config


def test_missing_variant_env_resolves_to_control(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(config.REWARD_VARIANT_ENV, raising=False)

    assert config._resolve_rewards() == config.BASE_REWARDS


def test_survival_rebalance_only_changes_its_two_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(config.REWARD_VARIANT_ENV, "survival_rebalance")

    resolved = config._resolve_rewards()

    assert resolved["SURVIVED_ROUND"] == 2.0
    assert resolved["WAITED"] == -0.3
    unchanged = {"SURVIVED_ROUND", "WAITED"}
    for key, value in config.BASE_REWARDS.items():
        if key not in unchanged:
            assert resolved[key] == value


def test_safety_bomb_adds_two_new_keys_without_touching_the_others(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(config.REWARD_VARIANT_ENV, "safety_bomb")

    resolved = config._resolve_rewards()

    assert resolved["SAFE_BOMB_PLACED"] == 0.2
    assert resolved["UNSAFE_BOMB_PLACED"] == -1.0
    for key, value in config.BASE_REWARDS.items():
        assert resolved[key] == value


def test_unknown_variant_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(config.REWARD_VARIANT_ENV, "unknown")

    with pytest.raises(ValueError, match="BOMBERMAN_DQN_REWARD_VARIANT must be one of"):
        config._resolve_rewards()


def test_control_is_undefined_for_the_bomb_safety_events() -> None:
    """SAFE_BOMB_PLACED/UNSAFE_BOMB_PLACED are tracked as diagnostics under
    every variant (train.py always emits one of the two on a confirmed bomb
    placement), but only "safety_bomb" gives either a nonzero reward --
    every other variant falls through reward_from_events'
    REWARDS.get(event, 0.0) default.
    """
    assert "SAFE_BOMB_PLACED" not in config.BASE_REWARDS
    assert "UNSAFE_BOMB_PLACED" not in config.BASE_REWARDS
