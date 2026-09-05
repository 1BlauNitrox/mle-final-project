"""Decision-rule tests for the preregistered Issue #46 analysis."""

from __future__ import annotations

from training.analyze_dqn_task2_experiment import (
    _comparisons,
    _criteria,
    _summarize,
)


def _row(
    *,
    treatment: str,
    model: str,
    scenario: str,
    world_seed: int,
    fraction: float,
) -> dict[str, object]:
    available = 50 if scenario != "classic" else 10
    coins = round(available * fraction)
    return {
        "treatment": treatment,
        "model": model,
        "scenario": scenario,
        "world_seed": world_seed,
        "collection_fraction": coins / available,
        "coins_collected": coins,
        "initially_available_coins": available,
        "episode_steps": 20,
        "survival_steps": 20,
        "survived": True,
        "coins_found": 1 if scenario == "classic" and treatment == "trained" else 0,
        "crates_destroyed": 1 if scenario == "classic" and treatment == "trained" else 0,
        "bombs_dropped": 1 if scenario == "classic" and treatment == "trained" else 0,
        "self_kills": 0,
        "invalid_actions": 0,
        "attempted_actions": 20,
        "action_bomb": 1 if scenario == "classic" and treatment == "trained" else 0,
        "action_up": 5,
        "action_right": 5,
        "action_down": 5,
        "action_left": 4,
        "action_wait": 0,
        "action_unknown": 0,
        "decision_time_p95_ms": 1.0,
        "decision_time_max_ms": 2.0,
    }


def test_registered_decision_rules_can_pass_on_compliant_evidence() -> None:
    rows: list[dict[str, object]] = []
    for seed in range(32_001, 32_041):
        for run in range(1, 6):
            rows.append(
                _row(
                    treatment="trained",
                    model=f"r{run}",
                    scenario="classic",
                    world_seed=seed,
                    fraction=0.4,
                )
            )
        rows.append(
            _row(
                treatment="untrained",
                model="untrained",
                scenario="classic",
                world_seed=seed,
                fraction=0.0,
            )
        )
    for seed in range(33_001, 33_041):
        for run in range(1, 6):
            rows.append(
                _row(
                    treatment="trained",
                    model=f"r{run}",
                    scenario="coin-heaven",
                    world_seed=seed,
                    fraction=0.96,
                )
            )
        rows.append(
            _row(
                treatment="task1",
                model="task1",
                scenario="coin-heaven",
                world_seed=seed,
                fraction=1.0,
            )
        )
    for treatment, models in (
        ("trained", [f"r{run}" for run in range(1, 6)]),
        ("untrained", ["untrained"]),
        ("task1", ["task1"]),
    ):
        for model in models:
            for seed in range(34_001, 34_041):
                rows.append(
                    _row(
                        treatment=treatment,
                        model=model,
                        scenario="loot-crate",
                        world_seed=seed,
                        fraction=0.1,
                    )
                )

    summaries = _summarize(rows)
    comparisons = _comparisons(rows)
    criteria = _criteria(rows, summaries, comparisons, deterministic=True)

    assert comparisons["classic_trained_minus_untrained"]["mean_difference"] == 0.4
    assert all(criteria.values())
