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

from training.hunting_curriculum import BombCredits, attack_exposure, hunting_layout


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
        if self.code_name == "DagobertDuckDQNAntiLoop":
            return original(self, event_name, *event_args)
        if not hasattr(self, "_issue225_rng"):
            slot = int(hashlib.sha256(self.agent_name.encode()).hexdigest()[:8], 16)
            words = np.random.SeedSequence([root_seed, slot, 217]).generate_state(3)
            numpy_seed = int(words[0])
            python_seed = (int(words[1]) << 32) | int(words[2])
            self._issue225_seed = numpy_seed, python_seed
            self._issue225_rng = (
                np.random.RandomState(numpy_seed).get_state(),
                random.Random(python_seed).getstate(),
            )
        numpy_seed, python_seed = self._issue225_seed
        previous = np.random.get_state(), random.getstate()
        numpy_seed_function, python_seed_function = np.random.seed, random.seed
        np.random.set_state(self._issue225_rng[0])
        random.setstate(self._issue225_rng[1])
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
            self._issue225_rng = np.random.get_state(), random.getstate()
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
    tracker=None,
    hunting_index=None,
    guard_mode=None,
    collect_head_training_trace=False,
    trace_stride=20,
    attack_head_name=None,
    collect_pursuit_training_trace=False,
    pursuit_head_name=None,
):

    source = (root / "source").resolve()
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    os.chdir(source)
    from agent_code.DagobertDuckDQNAntiLoop import callbacks, train
    from agents import AgentRunner
    from environment import BombeRLeWorld, WorldArgs

    actions = []
    executed_actions = []
    selector = callbacks.select_action
    original_act = callbacks.act
    original_build = BombeRLeWorld.build_arena
    original_action = BombeRLeWorld.perform_agent_action
    original_bombs = BombeRLeWorld.update_bombs
    original_hits = BombeRLeWorld.evaluate_explosions
    credits = BombCredits()
    observed = None
    layout_metadata = {}
    decision_record = None
    anchor_records = []
    placed_bomb_records = []
    pursuit_records = []

    def observe_action(world, agent, action):
        if agent is learner:
            executed_actions.append(action)
        before = set(world.bombs)
        result = original_action(world, agent, action)
        if agent is learner:
            for bomb in world.bombs:
                if bomb not in before:
                    credits.placed(
                        bomb, step=world.step, safe_attack=observed["safe_attack_position"]
                    )
                    if collect_head_training_trace:
                        if decision_record is None:
                            raise ValueError("Missing causal decision record for placed bomb")
                        placed_bomb_records.append(dict(decision_record))
        return result

    def observe_bombs(world):
        detonating = [bomb for bomb in world.bombs if bomb.timer <= 0]
        before = len(world.explosions)
        result = original_bombs(world)
        for bomb, explosion in zip(detonating, world.explosions[before:], strict=True):
            credits.detonated(bomb, explosion)
        return result

    def observe_hits(world):
        credits.observe_hits(world.explosions, world.active_agents)
        return original_hits(world)

    def build_arena(world):
        if hunting_index is None:
            return original_build(world)
        field, positions, metadata = hunting_layout(world_seed, hunting_index)
        layout_metadata.update(metadata)
        ordered = [world.agents[slot], *[a for a in world.agents if a is not world.agents[slot]]]
        for agent, position in zip(ordered, positions, strict=True):
            agent.x, agent.y = position
        return field, [], list(world.agents)

    def behavior(**kwargs):
        action = selector(**{**kwargs, "epsilon": epsilon})
        actions.append(action)
        return action

    def observed_act(policy, state):
        nonlocal decision_record
        if training and tracker is not None and hasattr(tracker, "observe"):
            tracker.observe(state)
        action = original_act(policy, state)
        if collect_head_training_trace or collect_pursuit_training_trace:
            import torch

            from agent_code.DagobertDuckDQNAntiLoop.endgame_pursuit_guard import (
                pursuit_input,
            )
            from agent_code.DagobertDuckDQNAntiLoop.legality import (
                framework_legal_action_mask,
            )
            from agent_code.DagobertDuckDQNAntiLoop.model import CPU_DEVICE
            from agent_code.DagobertDuckDQNAntiLoop.observations import observe
            from agent_code.DagobertDuckDQNAntiLoop.trapped_attack_guard import (
                TrappedAttackGuard,
            )

            features = observe(policy, state)
            if features is None:
                raise ValueError("Live game state produced no features")
            legal = framework_legal_action_mask(state)
            with torch.no_grad():
                q_values = (
                    policy.policy_network(torch.from_numpy(features).to(CPU_DEVICE)).cpu().numpy()
                )
            if collect_head_training_trace:
                trapped = (
                    TrappedAttackGuard().choose(
                        state,
                        "WAIT",
                        lambda: q_values,
                        legal,
                    )
                    == "BOMB"
                )
                decision_record = {
                    "step": int(state["step"]),
                    "features": features.tolist(),
                    "q_values": q_values.tolist(),
                    "legal": legal.tolist(),
                    "safe_attack": bool(observed and observed["safe_attack_position"]),
                    "trapped_attack": bool(trapped),
                    "selected_action": action,
                }
                if trace_stride <= 0:
                    raise ValueError("trace_stride must be positive")
                if int(state["step"]) % trace_stride == 0 or trapped:
                    anchor_records.append(dict(decision_record))
            if collect_pursuit_training_trace:
                from training.endgame_pursuit import pursuit_teacher_action

                teacher_action = pursuit_teacher_action(state, legal)
                if teacher_action is not None:
                    pursuit_records.append(
                        {
                            "step": int(state["step"]),
                            "input": pursuit_input(state, features, q_values).tolist(),
                            "teacher_action": teacher_action,
                            "selected_action": action,
                        }
                    )
        return action

    log_dir = root / "framework-logs" / str(os.getpid())
    log_dir.mkdir(parents=True, exist_ok=True)
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
        "issue225",
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
    if guard_mode is not None:
        env["BOMBERMAN_NARROW_LOOP_GUARD"] = guard_mode
    if attack_head_name is not None:
        env["BOMBERMAN_ATTACK_HEAD"] = attack_head_name
    if pursuit_head_name is not None:
        env["BOMBERMAN_PURSUIT_HEAD"] = pursuit_head_name
    with (
        release_agent_logs(),
        patch.dict(os.environ, env),
        patch.object(callbacks, "CHECKPOINT_PATH", checkpoint),
        patch.object(train, "CHECKPOINT_PATH", checkpoint),
        patch.object(callbacks, "_evaluation_checkpoint_path", lambda: checkpoint),
        patch.object(callbacks, "select_action", behavior),
        patch.object(callbacks, "act", observed_act),
        patch.object(BombeRLeWorld, "build_arena", build_arena),
        patch.object(BombeRLeWorld, "perform_agent_action", observe_action),
        patch.object(BombeRLeWorld, "update_bombs", observe_bombs),
        patch.object(BombeRLeWorld, "evaluate_explosions", observe_hits),
        patch.object(
            train, "_record_transition", tracker.record if tracker else train._record_transition
        ),
        patch.object(
            AgentRunner,
            "process_event",
            seeded_event_wrapper(AgentRunner.process_event, world_seed + 1_000_000),
        ),
    ):
        agents = [(name, False) for name in opponents]
        agents.insert(slot, ("DagobertDuckDQNAntiLoop", training))
        world = BombeRLeWorld(args, agents)
        learner = world.agents[slot]
        policy = learner.backend.runner.fake_self
        before_updates = policy.learner.update_steps if training else None
        world.new_round()
        # The framework ordinarily sets this in do_step before exposing state.
        world.user_input = "WAIT"
        initial_exposure = attack_exposure(world.get_state_for_agent(learner), search=True)
        exposure = {"safe_attack_steps": 0, "threatening_steps": 0, "safe_attack_bombs": 0}
        while world.running:
            alive = not learner.dead
            state_before = world.get_state_for_agent(learner) if alive else None
            observed = attack_exposure(state_before) if alive else None
            world.do_step()
            if not alive:
                continue
            exposure["safe_attack_steps"] += int(observed["safe_attack_position"])
            exposure["threatening_steps"] += int(observed["threatening_position"])
            exposure["safe_attack_bombs"] += int(
                observed["safe_attack_position"] and "BOMB_DROPPED" in learner.events
            )
            state = learner.last_game_state
            if state is not None:
                signature = board_coin_signature(state)
                rows.append(
                    {
                        "step": int(state["step"]),
                        "position": [int(value) for value in state["self"][3]],
                        "crates_left": int((state["field"] == 1).sum()),
                        "coins_visible": len(state["coins"]),
                        "score": int(state["self"][1]),
                        "opponents_left": len(state["others"]),
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
        bomb_credits = credits.snapshot()
        if collect_head_training_trace and len(placed_bomb_records) != len(bomb_credits):
            raise ValueError("Bomb training records and native credits are misaligned")
        if (
            sum(b["kill_credits"] for b in bomb_credits) != native["kills"]
            or sum(b["self_kill_credits"] for b in bomb_credits) != native["self_kills"]
        ):
            raise ValueError("Bomb-credit instrumentation disagrees with native metrics")
        result = {
            "world_seed": world_seed,
            "scenario": scenario,
            "hunting_layout": layout_metadata,
            "initial_exposure": initial_exposure,
            "attack_exposure": exposure,
            "bomb_credits": bomb_credits,
            "slot": slot,
            "epsilon": epsilon,
            "opponents": list(opponents),
            "native": native,
            "late_steps": rows,
            "actions_sha256": hashlib.sha256(json.dumps(actions).encode()).hexdigest(),
            "executed_actions_sha256": hashlib.sha256(
                json.dumps(executed_actions).encode()
            ).hexdigest(),
            "narrow_loop_guard": getattr(policy, "narrow_loop_guard", None).snapshot()
            if hasattr(policy, "narrow_loop_guard")
            else {
                "eligible": 0,
                "overrides": 0,
                "rejected_contested": 0,
                "rejected_no_candidate": 0,
            },
            "safe_attack_guard": getattr(policy, "safe_attack_guard", None).snapshot()
            if hasattr(policy, "safe_attack_guard")
            else {"eligible": 0, "overrides": 0, "rejected_rank": 0},
            "learned_attack_guard": getattr(policy, "learned_attack_guard", None).snapshot()
            if hasattr(policy, "learned_attack_guard")
            else {
                "eligible": 0,
                "overrides": 0,
                "rejected_rank": 0,
                "rejected_confidence": 0,
            },
            "endgame_pursuit_guard": getattr(policy, "endgame_pursuit_guard", None).snapshot()
            if hasattr(policy, "endgame_pursuit_guard")
            else {
                "eligible": 0,
                "overrides": 0,
                "rejected_confidence": 0,
                "rejected_bomb_safety": 0,
            },
            "completed_episodes": policy.completed_episodes,
            "loop_penalties": getattr(policy, "loop_penalty_count", 0),
            "optimizer_updates": policy.learner.update_steps if training else None,
            "optimizer_updates_this_episode": (
                policy.learner.update_steps - before_updates if training else None
            ),
        }
        if collect_head_training_trace:
            result["head_training_trace"] = {
                "anchors": anchor_records,
                "bombs": [
                    {**record, **credit}
                    for record, credit in zip(
                        placed_bomb_records,
                        bomb_credits,
                        strict=True,
                    )
                ],
            }
        if collect_pursuit_training_trace:
            result["pursuit_training_trace"] = pursuit_records
        return result
