"""End-to-end fixed-endpoint screening, including teacher and attack diagnostics."""

from copy import deepcopy

import pytest

from scripts.analyze_teacher_loop import analyze
from scripts.curriculum_io import read, write
from scripts.run_teacher_loop import CONFIG


def fixture(root):
    cfg = read(CONFIG)
    cfg["bootstrap_resamples"] = 20
    for setting in cfg["evaluation"]["final"].values():
        setting["seeds"][1] = setting["seeds"][0] + 1
    write(root / "config.json", cfg)
    for replica in (1, 2):
        pair = root / "pairs" / f"r{replica}"
        write(pair / "decision.json", {"status": "training_complete"})
        for arm in ("control", "memory", "reference"):
            directory = (
                pair / "reference/final"
                if arm == "reference"
                else pair / arm / "snapshots/final/final-evaluation"
            )
            for suite, setting in cfg["evaluation"]["final"].items():
                observations = []
                for i, seed in enumerate(range(setting["seeds"][0], setting["seeds"][1] + 1)):
                    observations.append(
                        {
                            "world_seed": seed,
                            "slot": i % (len(setting["opponents"]) + 1),
                            "opponents": setting["opponents"],
                            "scenario": setting["scenario"],
                            "epsilon": 0,
                            "native": {
                                "score": 3,
                                "kills": 0,
                                "survived": 1,
                                "self_kills": 0,
                                "coins": 3,
                                "initially_available_coins": 9,
                                "invalid": 0,
                                "decision_times_ms": [1.0, 2.0],
                            },
                            "loop_windows": {
                                "eligible": 100,
                                "looping": 50 if arm == "memory" else 100,
                            },
                            "attack_exposure": {
                                "safe_attack_steps": 2,
                                "threatening_steps": 3,
                                "safe_attack_bombs": 0,
                            },
                            "initial_exposure": {"reachable_attack_position": True},
                            "bomb_credits": [],
                            "loop_penalties": 1,
                        }
                    )
                write(directory / f"{suite}.json", {"rows": observations})
            if arm != "reference":
                write(
                    directory.parent / "state.json",
                    {
                        "episodes": 2,
                        "updates": cfg["updates"],
                        "rows": observations,
                        "tracker": {
                            "transitions": 2,
                            "generated": {"classic": 2},
                            "sampled": {"classic": 2},
                            "origins": ["classic", "classic"],
                            "teacher_counts": {
                                "generated_loop": 1,
                                "sampled_loop": 1,
                                "sampled_retention": 1,
                            },
                            "teacher_loss_updates": 10 if arm == "memory" else 0,
                            "teacher_loss_sum": 0.25 if arm == "memory" else 0,
                        },
                    },
                )
    return cfg


def test_complete_teacher_screen_and_retained_diagnostics(tmp_path):
    cfg = fixture(tmp_path)
    result = analyze(tmp_path)
    assert result["complete"] and result["eligible"]
    assert result["training"]["r1-memory"]["final"]["teacher_counts"]["sampled_loop"] == 1
    assert result["diagnostics"]["r1-memory"]["classic"]["attack"]["safe_attack_steps"] == 4
    final = tmp_path / "pairs/r1/memory/snapshots/final"
    path = final / "final-evaluation/classic.json"
    original = read(path)
    for failure in ("invalid", "loops", "score", "kills"):
        changed = deepcopy(original)
        for row in changed["rows"]:
            if failure == "loops":
                row["loop_windows"]["looping"] = 100
            else:
                row["native"][failure] += 1 if failure == "invalid" else -1
                if failure == "kills":
                    # No contradictory bomb-attribution data in this negative-metric fixture.
                    row.pop("bomb_credits")
        write(path, changed)
        second = tmp_path / "pairs/r2/memory/snapshots/final/final-evaluation/classic.json"
        if failure == "loops":
            write(second, changed)
        assert not analyze(tmp_path)["eligible"]
        write(second, original)
    write(path, original)
    state = read(final / "state.json")
    state["updates"] = cfg["updates"] - 1
    write(final / "state.json", state)
    with pytest.raises(ValueError, match="Incomplete fixed"):
        analyze(tmp_path)
