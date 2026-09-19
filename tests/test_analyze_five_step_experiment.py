from scripts.analyze_five_step_experiment import (
    hierarchical_interval,
    is_loop_window,
    loop_summary,
)


def loop_rows():
    return [
        {
            "step": index + 1,
            "position": [index % 2, 1],
            "crates_left": 0,
            "coins_visible": 0,
            "hazards": False,
            "progress": False,
            "board_coins_sha256": "fixed",
        }
        for index in range(24)
    ]


def test_loop_detector_requires_late_state_and_no_progress_or_hazards():
    assert is_loop_window(loop_rows())
    for change in ({"crates_left": 1}, {"coins_visible": 1}, {"hazards": True}, {"progress": True}):
        rows = loop_rows()
        rows[10].update(change)
        assert not is_loop_window(rows)


def test_loop_summary_reports_denominator_and_time_fraction():
    rows = loop_rows() + [{**loop_rows()[0], "crates_left": 1}]
    result = loop_summary([{"late_steps": rows}])
    assert result["late_state_steps"] == 24
    assert result["observed_steps"] == 25
    assert result["looping_windows"] == 1
    assert result["late_loop_rate"] == 1 / 24
    assert result["late_state_time_fraction"] == 24 / 25


def test_hierarchical_interval_keeps_replica_world_matrix():
    result = hierarchical_interval([[1, 1], [2, 2], [3, 3]], resamples=1000, seed=5)
    assert result["estimate"] == 2.0
    assert result["replica_means"] == [1.0, 2.0, 3.0]
    assert result["ci95"][0] <= 2.0 <= result["ci95"][1]
