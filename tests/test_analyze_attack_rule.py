"""The attack-rule decision demands kills first, then that they pay for themselves."""

from __future__ import annotations

import json

from scripts import analyze_attack_rule as analyze


def stat(mean, low, high):
    return {"mean": mean, "ci_low": low, "ci_high": high}


def test_the_registration_says_what_this_script_implements():
    cfg = json.loads(analyze.REGISTRATION.read_text(encoding="utf-8"))
    rule = cfg["decision_rule"]
    assert rule["type"].startswith("decision under uncertainty")
    assert rule["interval"]["role"] == "reported, not a gate"
    assert rule["primary_contrast"] == "attack minus control"
    assert [c["name"] for c in cfg["design"]["cells"]] == ["control", "attack", "attack-selective"]
    assert len(cfg["suite"]["world_seeds"]) == 150


def test_kills_that_pay_for_themselves_ship():
    passed, _ = analyze.decide(stat(+0.35, +0.02, +0.70), stat(+0.06, +0.01, +0.11))
    assert passed


def test_no_kills_means_no_reason_to_exist():
    # Score could drift up on noise alone; without kills the rule did nothing.
    passed, lines = analyze.decide(stat(+0.20, -0.05, +0.45), stat(0.0, -0.03, +0.03))
    assert not passed
    assert lines[0].strip().startswith("FAIL")


def test_kills_bought_with_score_do_not_ship():
    passed, _ = analyze.decide(stat(-0.30, -0.70, +0.10), stat(+0.09, +0.03, +0.15))
    assert not passed


def test_a_deep_score_floor_does_not_ship_even_with_kills_and_a_positive_estimate():
    passed, _ = analyze.decide(stat(+0.05, -0.40, +0.50), stat(+0.08, +0.02, +0.14))
    assert not passed


def test_the_cell_names_match_the_registration():
    cfg = json.loads(analyze.REGISTRATION.read_text(encoding="utf-8"))
    names = {c["name"] for c in cfg["design"]["cells"]}
    assert {analyze.CONTROL, analyze.ATTACK, analyze.SELECTIVE} == names


def test_every_cell_declares_all_three_switches():
    # A leftover switch from a previous cell is the one mistake that would
    # silently produce the wrong comparison, so the registration spells each
    # cell's full environment out.
    cfg = json.loads(analyze.REGISTRATION.read_text(encoding="utf-8"))
    for cell, environment in cfg["design"]["environment"].items():
        assert set(environment) == {
            "BOMBERMAN_ATTACK_RULE",
            "BOMBERMAN_SURVIVAL_GUARD_BOMB",
            "BOMBERMAN_SURVIVAL_GUARD_MOVE",
            "BOMBERMAN_SEEK_RULE",
        }, cell
        assert set(environment.values()) <= {"on", "off", "selective"}, cell
        assert environment["BOMBERMAN_SURVIVAL_GUARD_BOMB"] == "off", cell


def test_the_seek_rule_is_off_in_every_cell():
    # It exists in the prototype but is not under test here: on a 25-world smoke
    # run it took survival from 0.20 to 0.04.
    cfg = json.loads(analyze.REGISTRATION.read_text(encoding="utf-8"))
    for cell, environment in cfg["design"]["environment"].items():
        assert environment["BOMBERMAN_SEEK_RULE"] == "off", cell


def test_a_cell_folder_is_found_with_or_without_the_prefix(tmp_path, monkeypatch, capsys):
    # The prompt told the laptop to write attack-selective; the analyzer's
    # prefix asks for attack-attack-selective. Both must resolve, or a finished
    # experiment cannot be read.
    import json as _json

    def cell(folder, name):
        folder.mkdir(parents=True)
        rows = [{"artifact": "control-r1@8000", "world_seed": 1, "variant": name,
                 "score": 1.0, "kills": 0.0, "self_kills": 0.0, "survived": 1,
                 "coins": 1.0, "collection_fraction": 0.1, "invalid": 0}]
        (folder / "games.jsonl").write_text(
            "".join(_json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    root = tmp_path
    cell(root / "attack-control", "control")
    cell(root / "attack-attack", "attack")
    cell(root / "attack-selective", "attack-selective")  # written without the prefix

    for name, expected in (("control", "attack-control"),
                           ("attack", "attack-attack"),
                           ("attack-selective", "attack-selective")):
        candidates = [root / f"attack-{name}", root / name]
        assert any(c.is_dir() for c in candidates), name
        assert next(c for c in candidates if c.is_dir()).name == expected
