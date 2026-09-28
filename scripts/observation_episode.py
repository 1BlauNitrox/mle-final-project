"""Isolated single-process episode adapter; derived from the verified #207 runner."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import random
import sys
from unittest.mock import patch

import numpy as np


def board_coin_signature(state):
    payload = [state["field"].tolist(), sorted([int(x), int(y)] for x, y in state["coins"])]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


@contextlib.contextmanager
def release_agent_logs():
    before = {
        handler
        for logger in logging.Logger.manager.loggerDict.values()
        if isinstance(logger, logging.Logger)
        for handler in logger.handlers
    }
    try:
        yield
    finally:
        for logger in logging.Logger.manager.loggerDict.values():
            if not isinstance(logger, logging.Logger):
                continue
            for handler in list(logger.handlers):
                if isinstance(handler, logging.FileHandler) and handler not in before:
                    logger.removeHandler(handler)
                    handler.close()


def seeded_event_wrapper(original, root_seed):
    """Give every non-learning AgentRunner an isolated legacy RNG stream."""

    def process_event(self, event_name, *event_args):
        if self.code_name == "DagobertDuckDQNObservation":
            return original(self, event_name, *event_args)
        if not hasattr(self, "_issue211_rng"):
            slot = int(hashlib.sha256(self.agent_name.encode()).hexdigest()[:8], 16)
            words = np.random.SeedSequence([root_seed, slot, 211]).generate_state(3)
            numpy_seed = int(words[0])
            python_seed = (int(words[1]) << 32) | int(words[2])
            self._issue211_seed = numpy_seed, python_seed
            self._issue211_rng = (
                np.random.RandomState(numpy_seed).get_state(),
                random.Random(python_seed).getstate(),
            )
        numpy_seed, python_seed = self._issue211_seed
        previous = np.random.get_state(), random.getstate()
        numpy_seed_function, python_seed_function = np.random.seed, random.seed
        np.random.set_state(self._issue211_rng[0])
        random.setstate(self._issue211_rng[1])
        try:
            if event_name == "setup":
                np.random.seed = lambda value=None: numpy_seed_function(
                    numpy_seed if value is None else value
                )
                random.seed = lambda value=None, version=2: python_seed_function(
                    python_seed if value is None else value, version=version
                )
            return original(self, event_name, *event_args)
        finally:
            self._issue211_rng = np.random.get_state(), random.getstate()
            np.random.seed, random.seed = numpy_seed_function, python_seed_function
            np.random.set_state(previous[0])
            random.setstate(previous[1])

    return process_event


def play_episode(
    root,
    checkpoint,
    *,
    world_seed,
    opponents,
    training,
    epsilon,
    agent_seed,
    slot=0,
    scenario="classic",
):

    source = (root / "source").resolve()
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    os.chdir(source)
    from agent_code.DagobertDuckDQNObservation import callbacks, train
    from agents import AgentRunner
    from environment import BombeRLeWorld, WorldArgs

    actions = []
    selector = callbacks.select_action

    def behavior(**kwargs):
        action = selector(**{**kwargs, "epsilon": epsilon})
        actions.append(action)
        return action

    log_dir = root / "framework-logs"
    log_dir.mkdir(exist_ok=True)
    args = WorldArgs(
        True,
        30,
        False,
        0,
        False,
        None,
        False,
        not training,
        str(log_dir),
        False,
        "issue211",
        world_seed,
        False,
        scenario,
    )
    rows = []
    env = {
        "BOMBERMAN_AGENT_SEED": str(agent_seed),
        "BOMBERMAN_DQN_ACTION_MASKING": "framework_legal",
        "BOMBERMAN_DQN_ESCAPE_CONTINUATIONS": "on",
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "BOMBERMAN_COMPACT_LOGS": "1",
    }
    with (
        release_agent_logs(),
        patch.dict(os.environ, env),
        patch.object(callbacks, "CHECKPOINT_PATH", checkpoint),
        patch.object(train, "CHECKPOINT_PATH", checkpoint),
        patch.object(callbacks, "_evaluation_checkpoint_path", lambda: checkpoint),
        patch.object(callbacks, "select_action", behavior),
        patch.object(
            AgentRunner,
            "process_event",
            seeded_event_wrapper(AgentRunner.process_event, world_seed + 1_000_000),
        ),
    ):
        agents = [(name, False) for name in opponents]
        agents.insert(slot, ("DagobertDuckDQNObservation", training))
        world = BombeRLeWorld(args, agents)
        learner = world.agents[slot]
        policy = learner.backend.runner.fake_self
        before_updates = policy.learner.update_steps if training else None
        world.new_round()
        while world.running:
            alive = not learner.dead
            world.do_step()
            if not alive:
                continue
            state = learner.last_game_state
            if state is not None:
                signature = board_coin_signature(state)
                rows.append(
                    {
                        "step": int(state["step"]),
                        "position": [int(value) for value in state["self"][3]],
                        "crates_left": int((state["field"] == 1).sum()),
                        "coins_visible": len(state["coins"]),
                        "hazards": bool(state["bombs"] or np.any(state["explosion_map"])),
                        "progress": bool(
                            {"COIN_COLLECTED", "CRATE_DESTROYED", "KILLED_OPPONENT"}
                            & set(learner.events)
                        ),
                        "board_coins_sha256": signature,
                    }
                )
        world.end()
        native = world.round_statistics[world.round_id]["agents"][learner.name]
        return {
            "world_seed": world_seed,
            "slot": slot,
            "epsilon": epsilon,
            "opponents": list(opponents),
            "native": native,
            "late_steps": rows,
            "actions_sha256": hashlib.sha256(json.dumps(actions).encode()).hexdigest(),
            "completed_episodes": policy.completed_episodes,
            "optimizer_updates": policy.learner.update_steps if training else None,
            "optimizer_updates_this_episode": (
                policy.learner.update_steps - before_updates if training else None
            ),
        }
