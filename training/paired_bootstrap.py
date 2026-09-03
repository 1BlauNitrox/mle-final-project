"""Paired bootstrap comparison between two registered five-run experiments.

Issue #58 registers a paired comparison against the issue #41 baseline. Both
series evaluate five independently trained models on the same development world
seeds, so every episode of one series has exactly one partner in the other.

The resampling is two-stage because the two sources of variation are nested:
training runs differ from each other, and within one run the world seeds differ.
Resampling only seeds would treat five models as forty independent observations
and understate the interval; resampling only runs would ignore seed variation
entirely.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

BOOTSTRAP_RESAMPLES = 10_000
CONFIDENCE_PERCENTILES = (2.5, 97.5)


@dataclass(frozen=True)
class PairedComparison:
    """The registered comparison statistic and its uncertainty."""

    mean_difference: float
    ci_lower: float
    ci_upper: float
    resamples: int
    resampler_seed: int
    paired_models: int
    paired_seeds_per_model: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_primary_fractions(path: Path) -> dict[str, dict[int, float]]:
    """Read per-model, per-seed coin collection fractions from an evidence CSV.

    Only the primary pass is used. The repeat pass exists to verify determinism
    and would double-count every episode if included.
    """
    fractions: dict[str, dict[int, float]] = {}

    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["pass"] != "primary":
                continue

            available = int(row["initially_available_coins"])
            if available <= 0:
                raise ValueError(
                    f"Episode for {row['model']} seed {row['world_seed']} "
                    "reports no available coins"
                )

            model = row["model"]
            world_seed = int(row["world_seed"])
            if world_seed in fractions.setdefault(model, {}):
                raise ValueError(
                    f"Duplicate primary episode for {model} seed {world_seed}"
                )

            fractions[model][world_seed] = (
                int(row["coins_collected"]) / available
            )

    if not fractions:
        raise ValueError(f"No primary episodes found in {path}")

    return fractions


def paired_bootstrap(
    treatment: dict[str, dict[int, float]],
    baseline: dict[str, dict[int, float]],
    *,
    resampler_seed: int,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> PairedComparison:
    """Compare two experiments pairing on run index and world seed."""
    if resamples < 1:
        raise ValueError("resamples must be positive")

    models = sorted(set(treatment) & set(baseline))
    if not models:
        raise ValueError("The two experiments share no model index")
    if set(treatment) != set(baseline):
        raise ValueError(
            "Both experiments must contain the same model indices: "
            f"{sorted(treatment)} vs {sorted(baseline)}"
        )

    differences: list[np.ndarray] = []
    seed_counts: set[int] = set()
    for model in models:
        shared_seeds = sorted(set(treatment[model]) & set(baseline[model]))
        if not shared_seeds:
            raise ValueError(f"Model {model} shares no world seed")
        if set(treatment[model]) != set(baseline[model]):
            raise ValueError(
                f"Model {model} does not evaluate the same world seeds in both "
                "experiments"
            )

        seed_counts.add(len(shared_seeds))
        differences.append(
            np.asarray(
                [
                    treatment[model][seed] - baseline[model][seed]
                    for seed in shared_seeds
                ],
                dtype=float,
            )
        )

    if len(seed_counts) != 1:
        raise ValueError(
            f"Models are paired on differing seed counts: {sorted(seed_counts)}"
        )

    per_model = np.vstack(differences)
    mean_difference = float(per_model.mean())

    generator = np.random.default_rng(resampler_seed)
    model_count, seed_count = per_model.shape
    replicates = np.empty(resamples, dtype=float)

    for index in range(resamples):
        chosen_models = generator.integers(0, model_count, size=model_count)
        chosen_seeds = generator.integers(
            0,
            seed_count,
            size=(model_count, seed_count),
        )
        resampled = per_model[chosen_models[:, None], chosen_seeds]
        replicates[index] = resampled.mean()

    lower, upper = np.percentile(replicates, CONFIDENCE_PERCENTILES)

    return PairedComparison(
        mean_difference=mean_difference,
        ci_lower=float(lower),
        ci_upper=float(upper),
        resamples=resamples,
        resampler_seed=resampler_seed,
        paired_models=model_count,
        paired_seeds_per_model=seed_count,
    )
