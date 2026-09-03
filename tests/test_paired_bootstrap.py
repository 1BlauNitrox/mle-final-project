"""Tests for the registered paired bootstrap comparison."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from training.paired_bootstrap import (
    PairedComparison,
    paired_bootstrap,
    read_primary_fractions,
)

EVIDENCE_COLUMNS = (
    "model",
    "pass",
    "world_seed",
    "coins_collected",
    "initially_available_coins",
)


def _write_evidence(path: Path, rows: list[dict[str, object]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVIDENCE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _constant(models: int, seeds: int, fraction: float) -> dict:
    return {
        f"run-{model:02d}": {
            31_000 + seed: fraction for seed in range(1, seeds + 1)
        }
        for model in range(1, models + 1)
    }


def test_reads_only_the_primary_pass(tmp_path: Path) -> None:
    """The repeat pass verifies determinism and must not double-count."""
    rows = [
        {
            "model": "run-01",
            "pass": "primary",
            "world_seed": 31001,
            "coins_collected": 25,
            "initially_available_coins": 50,
        },
        {
            "model": "run-01",
            "pass": "repeat",
            "world_seed": 31001,
            "coins_collected": 25,
            "initially_available_coins": 50,
        },
    ]
    path = _write_evidence(tmp_path / "evidence.csv", rows)

    fractions = read_primary_fractions(path)

    assert fractions == {"run-01": {31001: 0.5}}


def test_duplicate_primary_episode_is_rejected(tmp_path: Path) -> None:
    row = {
        "model": "run-01",
        "pass": "primary",
        "world_seed": 31001,
        "coins_collected": 10,
        "initially_available_coins": 50,
    }
    path = _write_evidence(tmp_path / "evidence.csv", [row, dict(row)])

    with pytest.raises(ValueError, match="Duplicate primary episode"):
        read_primary_fractions(path)


def test_constant_difference_has_a_degenerate_interval() -> None:
    """Analytic case: every resample of a constant has the same mean."""
    treatment = _constant(5, 40, 0.90)
    baseline = _constant(5, 40, 0.80)

    result = paired_bootstrap(treatment, baseline, resampler_seed=7)

    assert result.mean_difference == pytest.approx(0.10)
    assert result.ci_lower == pytest.approx(0.10)
    assert result.ci_upper == pytest.approx(0.10)
    assert result.paired_models == 5
    assert result.paired_seeds_per_model == 40
    assert result.resamples == 10_000


def test_interval_brackets_the_point_estimate_for_varying_data() -> None:
    treatment = {
        "run-01": {31001: 1.0, 31002: 0.9, 31003: 0.8},
        "run-02": {31001: 0.7, 31002: 0.6, 31003: 1.0},
    }
    baseline = {
        "run-01": {31001: 0.5, 31002: 0.5, 31003: 0.5},
        "run-02": {31001: 0.5, 31002: 0.5, 31003: 0.5},
    }

    result = paired_bootstrap(treatment, baseline, resampler_seed=1)

    assert result.ci_lower < result.mean_difference < result.ci_upper


def test_result_is_reproducible_for_a_fixed_seed() -> None:
    """The registered seed must make the interval exactly reproducible.

    A second seed is compared on a large enough grid that the achievable
    resample means are not so coarse that two seeds trivially collide.
    """
    treatment = {
        f"run-{model:02d}": {
            seed: (model * 7 + seed * 3) % 11 / 10 for seed in range(1, 21)
        }
        for model in range(1, 6)
    }
    baseline = {
        f"run-{model:02d}": {seed: 0.5 for seed in range(1, 21)}
        for model in range(1, 6)
    }

    first = paired_bootstrap(treatment, baseline, resampler_seed=42)
    second = paired_bootstrap(treatment, baseline, resampler_seed=42)
    other = paired_bootstrap(treatment, baseline, resampler_seed=43)

    assert first == second
    assert first.mean_difference == pytest.approx(other.mean_difference)
    assert first.ci_lower != other.ci_lower


def test_two_stage_interval_is_wider_than_ignoring_run_variation() -> None:
    """Between-run spread must widen the interval, not be averaged away."""
    spread = {
        "run-01": {1: 1.0, 2: 1.0, 3: 1.0},
        "run-02": {1: 0.0, 2: 0.0, 3: 0.0},
    }
    uniform = {
        "run-01": {1: 0.5, 2: 0.5, 3: 0.5},
        "run-02": {1: 0.5, 2: 0.5, 3: 0.5},
    }
    baseline = {
        "run-01": {1: 0.5, 2: 0.5, 3: 0.5},
        "run-02": {1: 0.5, 2: 0.5, 3: 0.5},
    }

    spread_result = paired_bootstrap(spread, baseline, resampler_seed=3)
    uniform_result = paired_bootstrap(uniform, baseline, resampler_seed=3)

    assert spread_result.mean_difference == pytest.approx(0.0)
    assert uniform_result.mean_difference == pytest.approx(0.0)
    spread_width = spread_result.ci_upper - spread_result.ci_lower
    uniform_width = uniform_result.ci_upper - uniform_result.ci_lower
    assert spread_width > uniform_width


def test_mismatched_model_indices_are_rejected() -> None:
    with pytest.raises(ValueError, match="same model indices"):
        paired_bootstrap(
            {"run-01": {1: 0.5}, "run-02": {1: 0.5}},
            {"run-01": {1: 0.5}},
            resampler_seed=0,
        )


def test_mismatched_world_seeds_are_rejected() -> None:
    with pytest.raises(ValueError, match="same world seeds"):
        paired_bootstrap(
            {"run-01": {1: 0.5, 2: 0.5}},
            {"run-01": {1: 0.5, 3: 0.5}},
            resampler_seed=0,
        )


def test_unequal_seed_counts_across_models_are_rejected() -> None:
    with pytest.raises(ValueError, match="differing seed counts"):
        paired_bootstrap(
            {"run-01": {1: 0.5, 2: 0.5}, "run-02": {1: 0.5}},
            {"run-01": {1: 0.5, 2: 0.5}, "run-02": {1: 0.5}},
            resampler_seed=0,
        )


def test_comparison_serialises_for_the_result_record() -> None:
    result = PairedComparison(
        mean_difference=0.1,
        ci_lower=0.05,
        ci_upper=0.15,
        resamples=10_000,
        resampler_seed=58,
        paired_models=5,
        paired_seeds_per_model=40,
    )

    assert result.as_dict()["resampler_seed"] == 58
    assert set(result.as_dict()) == {
        "mean_difference",
        "ci_lower",
        "ci_upper",
        "resamples",
        "resampler_seed",
        "paired_models",
        "paired_seeds_per_model",
    }
