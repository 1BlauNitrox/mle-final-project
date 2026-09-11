"""Seeded opponent streams survive entropy setup and callback interleaving."""

import random
from types import SimpleNamespace

import numpy as np
import pytest

from training.seeded_framework import seeds, wrap_process_event


def draw(runner, event):
    if event == "setup":
        np.random.seed()
        random.seed()
    if event == "fail":
        raise RuntimeError("callback failed")
    return int(np.random.randint(1000000)), random.randrange(1000000)


def test_setup_entropy_requests_and_interleaving_are_reproducible():
    wrapped = wrap_process_event(draw, 109101, {"peaceful_agent": 1, "coin_collector_agent": 2})
    a, b = SimpleNamespace(code_name="peaceful_agent"), SimpleNamespace(code_name="peaceful_agent")
    other = SimpleNamespace(code_name="coin_collector_agent")
    assert wrapped(a, "setup") == wrapped(b, "setup")
    left = [wrapped(a, "act") for _ in range(20)]
    right = []
    for _ in range(20):
        wrapped(other, "act")
        np.random.random(17)
        random.random()
        right.append(wrapped(b, "act"))
    assert left == right
    assert seeds(1, 1) != seeds(1, 2) != seeds(2, 2)


def test_global_randomness_and_seed_functions_are_restored_on_exception():
    wrapped = wrap_process_event(draw, 1, {"peaceful_agent": 1})
    np.random.seed(42)
    random.seed(42)
    before_np, before_py = np.random.get_state(), random.getstate()
    np_seed, py_seed = np.random.seed, random.seed
    with pytest.raises(RuntimeError):
        wrapped(SimpleNamespace(code_name="peaceful_agent"), "fail")
    after = np.random.get_state()
    assert before_np[0] == after[0] and np.array_equal(before_np[1], after[1])
    assert before_np[2:] == after[2:] and before_py == random.getstate()
    assert np.random.seed is np_seed and random.seed is py_seed


def test_observed_learned_agent_is_not_wrapped():
    calls = []
    wrapped = wrap_process_event(lambda *args: calls.append(args), 1, {"peaceful_agent": 1})
    learner = SimpleNamespace(code_name="DagobertDuckDQNTask3")
    wrapped(learner, "setup")
    assert calls == [(learner, "setup")]
    assert not hasattr(learner, "_task3_rng")


@pytest.mark.parametrize(
    "agent_seed, opponents",
    [
        (None, ["peaceful_agent"]),
        (-1, []),
        (1, ["random_agent"]),
        (1, ["peaceful_agent", "peaceful_agent"]),
    ],
)
def test_runner_rejects_uncontrolled_opponent_configuration(tmp_path, agent_seed, opponents):
    from training.run_experiment import run_experiment

    output = tmp_path / "outputs"
    with pytest.raises(ValueError, match="Seeded opponents"):
        run_experiment(
            agent="DagobertDuckDQNTask3",
            mode="evaluation",
            scenario="classic",
            rounds=1,
            world_seed=1,
            agent_seed=agent_seed,
            opponents=opponents,
            output_root=output,
            seed_opponents=True,
        )
    assert not output.exists()
