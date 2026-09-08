"""Contract tests for the prospective Issue #107 factorial campaign."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from training.analyze_issue107_task2_factorial import (
    _bootstrap_result,
    _eligibility,
    _select_cell,
    _select_representative,
    _validate_registered_plan,
)
from training.run_issue107_campaign import (
    PLAN_PATHS,
    CampaignLimits,
    CampaignResourceLimitExceeded,
    CampaignResourceMonitor,
    main,
    validate_protocol,
)
from training.run_plan import load_plan


def _comparison(mean: float, lower: float) -> dict[str, float]:
    return {
        "mean_difference": mean,
        "bonferroni_98_75_lower": lower,
        "ci95_lower": lower,
    }


def _guards(lower: float = 0.0) -> dict[str, dict[str, float]]:
    return {f"scenario-{index}": {"ci95_lower": lower} for index in range(6)}


def _summaries() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for cell, classic, coin, self_kill in (
        ("A", 0.20, 0.50, 0.40),
        ("B", 0.30, 0.51, 0.20),
        ("C", 0.28, 0.70, 0.35),
        ("D", 0.35, 0.72, 0.18),
    ):
        for index in range(1, 6):
            rows.extend(
                [
                    {
                        "cell": cell,
                        "model": f"r{index}",
                        "scenario": "classic",
                        "mean_collection_fraction": classic + index / 1000,
                        "self_kill_rate": self_kill,
                    },
                    {
                        "cell": cell,
                        "model": f"r{index}",
                        "scenario": "coin-heaven",
                        "mean_collection_fraction": coin,
                        "self_kill_rate": 0.0,
                    },
                ]
            )
    return rows


class _FakeProcess:
    def __init__(
        self,
        pid: int,
        *,
        cpu_seconds: float = 0.0,
        memory_bytes: int = 0,
        children: tuple[_FakeProcess, ...] = (),
    ) -> None:
        self.pid = pid
        self.cpu_seconds = cpu_seconds
        self.memory_bytes = memory_bytes
        self._children = children
        self.terminated = False
        self.killed = False

    def children(self, *, recursive: bool) -> list[_FakeProcess]:
        assert recursive
        return list(self._children)

    def cpu_times(self) -> SimpleNamespace:
        return SimpleNamespace(user=self.cpu_seconds, system=0.0)

    def memory_info(self) -> SimpleNamespace:
        return SimpleNamespace(rss=self.memory_bytes)

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


def _monitor(tmp_path, process: _FakeProcess, clock: list[float], limits: CampaignLimits):
    return CampaignResourceMonitor(
        state_path=tmp_path / "resources.json",
        authorized_at="1970-01-01T00:00:00Z",
        limits=limits,
        time_fn=lambda: clock[0],
        process_factory=lambda pid: process,
        wait_procs=lambda processes, timeout: (list(processes), []),
    )


def test_protocol_is_complete_and_uses_disjoint_registered_seeds() -> None:
    report = validate_protocol()

    assert report["plans"] == {
        "A": "issue107-cell-a-control",
        "B": "issue107-cell-b-escape",
        "C": "issue107-cell-c-replay",
        "D": "issue107-cell-d-combined",
        "untrained": "issue107-untrained",
        "frozen_task1": "issue107-frozen-task1",
    }
    assert report["training_replicas"] == 20
    assert report["training_episodes"] == 200_000
    assert report["evaluation_episodes"] == 5_120
    assert report["compute_authorized"] is False
    assert report["scientific_result"] is None


def test_campaign_refuses_execution_without_explicit_authorization(capsys) -> None:
    assert main([]) == 1
    assert "--authorize-compute" in capsys.readouterr().err


def test_campaign_wall_limit_terminates_registered_processes(tmp_path) -> None:
    clock = [0.0]
    process = _FakeProcess(11)
    monitor = _monitor(
        tmp_path,
        process,
        clock,
        CampaignLimits(cpu_seconds=100.0, wall_seconds=1.0, memory_bytes=100),
    )
    monitor.register(process.pid)

    clock[0] = 1.01
    with pytest.raises(CampaignResourceLimitExceeded, match="wall ceiling"):
        monitor.check()

    assert process.terminated


def test_campaign_memory_limit_is_aggregate_across_process_tree(tmp_path) -> None:
    child = _FakeProcess(12, memory_bytes=60)
    process = _FakeProcess(11, memory_bytes=50, children=(child,))
    monitor = _monitor(
        tmp_path,
        process,
        [0.0],
        CampaignLimits(cpu_seconds=100.0, wall_seconds=100.0, memory_bytes=100),
    )

    with pytest.raises(CampaignResourceLimitExceeded, match="memory ceiling"):
        monitor.register(process.pid)

    assert process.terminated
    assert child.terminated


def test_campaign_resume_counts_cpu_from_completed_jobs(tmp_path) -> None:
    clock = [0.0]
    first = _FakeProcess(11, cpu_seconds=0.6)
    limits = CampaignLimits(cpu_seconds=1.0, wall_seconds=100.0, memory_bytes=100)
    monitor = _monitor(tmp_path, first, clock, limits)
    monitor.register(first.pid)
    monitor.unregister(first.pid)

    second = _FakeProcess(22, cpu_seconds=0.5)
    resumed = _monitor(tmp_path, second, clock, limits)
    with pytest.raises(CampaignResourceLimitExceeded, match="CPU ceiling"):
        resumed.register(second.pid)

    assert second.terminated


@pytest.mark.parametrize("mutated_field", ["seed", "source", "parent"])
def test_analyzer_rejects_outputs_not_bound_to_registered_plan(mutated_field) -> None:
    resolved = load_plan(PLAN_PATHS["A"]).to_dict()
    _validate_registered_plan("A", resolved)
    mutated = copy.deepcopy(resolved)
    if mutated_field == "seed":
        mutated["replicas"][0]["world_seed"] += 1
    elif mutated_field == "source":
        mutated["fingerprints"]["source"] = "0" * 64
    else:
        mutated["fingerprints"]["parent_artifacts"]["r1"] = "0" * 64

    with pytest.raises(ValueError, match="does not match registered plan"):
        _validate_registered_plan("A", mutated)


def test_bootstrap_reports_registered_and_multiplicity_adjusted_intervals() -> None:
    result = _bootstrap_result(np.full((5, 40), 0.2), resampler_seed=107)

    assert result["mean_difference"] == pytest.approx(0.2)
    assert result["ci95_lower"] == pytest.approx(0.2)
    assert result["bonferroni_98_75_lower"] == pytest.approx(0.2)
    assert result["paired_models"] == 5
    assert result["paired_seeds_per_model"] == 40
    assert result["resamples"] == 10_000


def test_treatment_eligibility_requires_efficacy_and_every_guard() -> None:
    comparisons = {
        "escape_b_minus_a": _comparison(0.15, 0.01),
        "escape_d_minus_c": _comparison(0.16, 0.01),
        "replay_c_minus_a": _comparison(0.10, 0.01),
        "replay_d_minus_b": _comparison(0.11, 0.01),
        "guards_b_minus_a": _guards(),
        "guards_c_minus_a": _guards(),
        "guards_d_minus_c": _guards(),
        "guards_d_minus_b": _guards(),
    }

    assert _eligibility(comparisons) == {"A": True, "B": True, "C": True, "D": True}

    comparisons["guards_d_minus_b"] = _guards(lower=-0.05)
    assert _eligibility(comparisons)["D"] is False


def test_no_eligible_treatment_falls_back_to_control() -> None:
    gates = {
        cell: {
            "overall_passed": False,
            "task2_passed": False,
            "task1_passed": False,
        }
        for cell in "ABCD"
    }
    eligibility = {"A": True, "B": False, "C": False, "D": False}

    assert _select_cell(eligibility, gates, _summaries()) == "A"


def test_eligible_cells_use_preregistered_lexicographic_order() -> None:
    gates = {
        "A": {"overall_passed": False, "task2_passed": False, "task1_passed": False},
        "B": {"overall_passed": False, "task2_passed": True, "task1_passed": False},
        "C": {"overall_passed": False, "task2_passed": False, "task1_passed": True},
        "D": {"overall_passed": True, "task2_passed": True, "task1_passed": True},
    }
    eligibility = {"A": True, "B": True, "C": True, "D": True}

    assert _select_cell(eligibility, gates, _summaries()) == "D"


def test_representative_is_median_classic_replica_not_best_seed() -> None:
    assert _select_representative("D", _summaries()) == "r3"
