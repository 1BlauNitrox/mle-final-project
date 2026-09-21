"""End-to-end hard-gate coverage for learned attack analysis."""

import json

from scripts.analyze_learned_attack_gate import gate


def row(*, kills=0, overrides=0, coins=1):
    return {
        "native": {
            "score": kills * 5,
            "kills": kills,
            "survived": 1,
            "self_kills": 0,
            "invalid": 0,
            "coins": coins,
            "initially_available_coins": 1,
            "decision_times_ms": [1.0],
        },
        "broad_loop_windows": {"eligible": 10, "looping": 0},
        "learned_attack_guard": {
            "eligible": overrides,
            "overrides": overrides,
            "rejected_rank": 0,
            "rejected_confidence": 0,
        },
    }


def test_attack_exposure_is_a_hard_gate_alongside_retention_metrics(tmp_path):
    screen = {
        "kills_gain": 0.025,
        "score_gain": 0.0,
        "survival_loss": 0.05,
        "self_kills_increase": 0.05,
        "collection_loss": 0.05,
        "loop_ratio": 1.1,
        "loop_absolute_rate": 0.02,
        "minimum_attack_overrides": 3,
        "multiplayer_suites": ["classic", "mixed", "peaceful"],
        "loop_suites": ["classic", "mixed", "peaceful"],
        "invalid_actions": {
            "pooled_increase_per_game": 0.25,
            "block_increase_per_game": 0.5,
            "solo_maximum_total": 0,
        },
    }
    cfg = {"screen": screen}
    for arm in ("baseline", "rank2"):
        for suite in ("classic", "mixed", "peaceful", "coins", "crates"):
            path = tmp_path / "results" / "pilot" / arm / f"{suite}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            candidate_kill = int(arm == "rank2" and suite == "classic")
            candidate_overrides = int(arm == "rank2" and suite == "classic") * 3
            path.write_text(
                json.dumps(
                    {
                        "rows": [
                            row(
                                kills=candidate_kill,
                                overrides=candidate_overrides,
                            )
                        ]
                    }
                ),
                encoding="utf-8",
            )
    result = gate(tmp_path, cfg, "pilot", "rank2")
    assert result["passed"]
    assert result["effects"]["attack_overrides"] == 3

    classic = tmp_path / "results/pilot/rank2/classic.json"
    payload = json.loads(classic.read_text(encoding="utf-8"))
    payload["rows"][0]["learned_attack_guard"]["overrides"] = 0
    classic.write_text(json.dumps(payload), encoding="utf-8")
    result = gate(tmp_path, cfg, "pilot", "rank2")
    assert not result["passed"]
    assert not result["gates"]["attack_exposure"]
