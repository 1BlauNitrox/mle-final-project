"""A process that plays thousands of episodes must not run out of descriptors."""

from __future__ import annotations

import logging

from scripts.pilot_task4_competition import released_agent_log_handlers


def agent_round(tmp_path, index):
    """What the framework does when an agent starts: attach a fresh file handler."""
    opened = []
    for suffix in ("_wrapper", "_code"):
        logger = logging.getLogger(f"leak_probe_agent{suffix}")
        handler = logging.FileHandler(tmp_path / f"round-{index}{suffix}.log", mode="w")
        logger.addHandler(handler)
        opened.append(handler)
    return opened


def test_handlers_opened_by_a_round_are_closed_when_it_ends(tmp_path):
    with released_agent_log_handlers():
        opened = agent_round(tmp_path, 0)
        assert all(handler.stream is not None for handler in opened)
    assert all(handler.stream is None for handler in opened), "handler left open"
    for suffix in ("_wrapper", "_code"):
        assert not logging.getLogger(f"leak_probe_agent{suffix}").handlers


def test_many_rounds_do_not_accumulate_handlers(tmp_path):
    # The line-up rehearsal died between episodes 1,425 and 1,625 with
    # OSError: [Errno 24]. Without the fix this logger would hold 2 x rounds
    # handlers; the ceiling has to be flat, not merely higher.
    for index in range(50):
        with released_agent_log_handlers():
            agent_round(tmp_path, index)
        for suffix in ("_wrapper", "_code"):
            assert len(logging.getLogger(f"leak_probe_agent{suffix}").handlers) == 0


def test_handlers_that_predate_the_round_are_left_alone(tmp_path):
    # Only what a round opened is closed; a caller's own logging survives.
    logger = logging.getLogger("leak_probe_agent_code")
    outer = logging.FileHandler(tmp_path / "outer.log", mode="w")
    logger.addHandler(outer)
    try:
        with released_agent_log_handlers():
            agent_round(tmp_path, 99)
        assert outer in logger.handlers and outer.stream is not None
    finally:
        logger.removeHandler(outer)
        outer.close()


def test_handlers_are_released_even_when_the_round_raises(tmp_path):
    opened = []
    try:
        with released_agent_log_handlers():
            opened = agent_round(tmp_path, 1)
            raise RuntimeError("round failed")
    except RuntimeError:
        pass
    assert all(handler.stream is None for handler in opened)
