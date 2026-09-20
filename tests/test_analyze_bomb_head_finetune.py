from __future__ import annotations

import json

from scripts.analyze_bomb_head_finetune import analyze


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def row(*, score=0, kills=0, survived=1, self_kills=0, invalid=0, coins=1):
    return {
        "native": {
            "score": score,
            "kills": kills,
            "survived": survived,
            "self_kills": self_kills,
            "invalid": invalid,
            "coins": coins,
            "initially_available_coins": 2,
            "decision_times_ms": [1.0, 2.0],
        },
        "broad_loop_windows": {"eligible": 10, "looping": 1},
    }


def test_analyzer_selects_only_candidate_passing_all_registered_gates(tmp_path):
    suites = {name: {} for name in ("classic", "mixed", "peaceful", "coins", "crates")}
    config = {
        "training": {"snapshot_steps": [25, 100]},
        "evaluation": {"pilot": suites},
        "screen": {
            "kills_gain": 0.025,
            "score_gain": 0.0,
            "survival_loss": 0.05,
            "self_kills_increase": 0.05,
            "collection_loss": 0.05,
            "loop_ratio": 1.1,
            "loop_absolute_rate": 0.02,
            "multiplayer_suites": ["classic", "mixed", "peaceful"],
            "loop_suites": ["classic", "mixed", "peaceful"],
            "invalid_actions": {
                "pooled_increase_per_game": 0.25,
                "block_increase_per_game": 0.5,
                "solo_maximum_total": 0,
            },
            "latency_p95_ms": 50,
            "latency_max_ms": 100,
        },
    }
    write(tmp_path / "config.json", config)
    for suite in suites:
        write(tmp_path / "results/pilot/baseline" / f"{suite}.json", {"rows": [row()]})
        passing = row(score=1, kills=1) if suite not in {"coins", "crates"} else row()
        write(tmp_path / "results/pilot/u25" / f"{suite}.json", {"rows": [passing]})
        failing = row(score=-1) if suite not in {"coins", "crates"} else row()
        write(tmp_path / "results/pilot/u100" / f"{suite}.json", {"rows": [failing]})

    report = analyze(tmp_path)
    assert report["complete"]
    assert report["eligible"] == ["u25"]
    assert report["selected"] == "u25"
    assert report["pilot"]["u25"]["passed"]
    assert not report["pilot"]["u100"]["passed"]
