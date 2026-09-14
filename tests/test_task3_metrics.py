"""Match metrics retain attribution, ties and missing historical observations."""

import pytest

from tests.test_experiment_metrics import make_agent_statistics
from training.aggregate import read_episodes_csv
from training.metrics import normalize_episode_rows, write_episodes_csv


def stats(**agents):
    return {"by_round": {"1": {"agents": agents}}}


def test_multiplayer_scores_and_kills_roundtrip(tmp_path):
    rows = normalize_episode_rows(stats(
        learner=make_agent_statistics(score=7, kills=1),
        opponent=make_agent_statistics(score=2, kills=0),
    ), "evaluation")
    path = tmp_path / "episodes.csv"
    write_episodes_csv(rows, path)
    learner, opponent = read_episodes_csv(path)
    assert learner["opponents_eliminated"] == 1
    assert opponent["opponents_eliminated"] == 0
    assert learner["score_margin"] == 5
    assert opponent["score_margin"] == -5
    assert learner["first_place"] == 1 and opponent["first_place"] == 0
    assert learner["opponent_count"] == 1


def test_ties_are_separate_from_first_place_and_lower_ties_are_not_first():
    rows = normalize_episode_rows(stats(
        a=make_agent_statistics(score=4, kills=0),
        b=make_agent_statistics(score=4, kills=0),
        c=make_agent_statistics(score=0, kills=0),
        d=make_agent_statistics(score=0, kills=0),
    ), "evaluation")
    assert [r["first_place"] for r in rows] == [0, 0, 0, 0]
    assert [r["tied_first"] for r in rows] == [1, 1, 0, 0]
    assert all(r["opponent_count"] == 3 for r in rows)


def test_missing_kills_and_opponent_free_match_are_unavailable():
    row, = normalize_episode_rows(stats(a=make_agent_statistics()), "evaluation")
    assert row["opponents_eliminated"] is None
    assert row["first_place"] is row["score_margin"] is row["tied_first"] is None
    assert row["opponent_count"] == 0


@pytest.mark.parametrize("kills", [-1, True, 0.5])
def test_invalid_kill_count_fails(kills):
    with pytest.raises(ValueError, match="kills"):
        normalize_episode_rows(stats(a=make_agent_statistics(kills=kills)), "evaluation")
