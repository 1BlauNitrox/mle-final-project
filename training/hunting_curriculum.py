"""Training-only start states and transparent replay/update instrumentation."""

from __future__ import annotations

from collections import Counter, deque
from copy import deepcopy

import numpy as np

DIRECTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))


def distances(field, start, blocked=()):
    blocked = set(blocked)
    result, queue = {start: 0}, deque([start])
    while queue:
        x, y = queue.popleft()
        for dx, dy in DIRECTIONS:
            p = x + dx, y + dy
            if (
                0 <= p[0] < field.shape[0]
                and 0 <= p[1] < field.shape[1]
                and field[p] == 0
                and p not in blocked
                and p not in result
            ):
                result[p] = result[x, y] + 1
                queue.append(p)
    return result


def escape_at(field, position, opponents, bombs=(), explosion_map=None):
    from agent_code.DagobertDuckDQNTask3.features.bombs_and_crates import (
        build_danger_map,
        safe_escape_exists,
    )

    # Match the policy's convention: placement is decremented by the framework
    # before the next observable state, so hypothetical timer = BOMB_TIMER - 1.
    proposed = [*bombs, (position, 3)]
    explosion_map = np.zeros_like(field) if explosion_map is None else explosion_map
    danger = build_danger_map(field, proposed, explosion_map)
    return safe_escape_exists(field, danger, set(opponents), proposed, position)


def attack_exposure(state, *, search=False):
    """Public-state geometric diagnostics; never supplied to the policy."""
    from agent_code.DagobertDuckDQNTask3.features.bombs_and_crates import blast_footprint

    field, position = state["field"], tuple(state["self"][3])
    opponents = {tuple(o[3]) for o in state["others"]}
    ready = bool(state["self"][2])
    threatened = bool(opponents.intersection(blast_footprint(position, field)))
    safe = (
        ready
        and threatened
        and escape_at(field, position, opponents, state["bombs"], state["explosion_map"])
    )
    result = {"threatening_position": threatened, "safe_attack_position": bool(safe)}
    if search:
        reachable = distances(field, position, opponents | {tuple(b[0]) for b in state["bombs"]})
        options = [p for p in reachable if opponents.intersection(blast_footprint(p, field))]
        valid = [p for p in options if escape_at(field, p, opponents)]
        result["reachable_attack_position"] = bool(valid)
        result["attack_path_distance"] = min((reachable[p] for p in valid), default=None)
    return result


def hunting_layout(seed, index):
    """17x17 official walls; open or crate corridors; no coins or initial hazards.

    Rejection sampling uses only its local seed. Every accepted spawn has two
    free neighbours, a reachable opponent, and a reachable escapable attack tile.
    """
    rng = np.random.default_rng(seed)
    corridor = (index // 2) % 2 == 1
    desired_distance = (2, 3, 5, 7)[(index // 4) % 4]
    field = np.zeros((17, 17), dtype=int)
    field[[0, -1], :] = -1
    field[:, [0, -1]] = -1
    field[2:16:2, 2:16:2] = -1
    if corridor:
        field[(rng.random(field.shape) < 0.35) & (field == 0)] = 1
    # Connected central routes with side exits, preserving every official wall.
    for x in range(3, 14):
        for y in range(3, 14):
            if (x in (5, 7, 9, 11) or y in (5, 7, 9, 11)) and field[x, y] != -1:
                field[x, y] = 0
    candidates = [
        tuple(map(int, p)) for p in np.argwhere(field == 0) if 3 <= p[0] <= 13 and 3 <= p[1] <= 13
    ]
    candidates = [candidates[int(i)] for i in rng.permutation(len(candidates))]
    for start in candidates:
        reach = distances(field, start)
        targets = [p for p in candidates if reach.get(p) == desired_distance]
        for target in targets:
            positions = [start, target]
            for p in ((1, 1), (15, 15)):
                field[p] = 0
                for dx, dy in DIRECTIONS:
                    q = p[0] + dx, p[1] + dy
                    if field[q] == 1:
                        field[q] = 0
                positions.append(p)
            if any(
                sum(
                    field[p[0] + dx, p[1] + dy] == 0 and (p[0] + dx, p[1] + dy) not in positions
                    for dx, dy in DIRECTIONS
                )
                < 2
                for p in positions
            ):
                continue
            state = {
                "field": field,
                "self": ("learner", 0, True, start),
                "others": [(str(i), 0, True, p) for i, p in enumerate(positions[1:])],
                "bombs": [],
                "explosion_map": np.zeros_like(field),
            }
            exposure = attack_exposure(state, search=True)
            if exposure["reachable_attack_position"]:
                return (
                    field,
                    positions,
                    {
                        "layout": "corridor" if corridor else "open",
                        "opponent_path_distance": desired_distance,
                        **exposure,
                    },
                )
    raise ValueError(f"No valid hunting layout for seed {seed}")


def training_setting(arm, index):
    hunting = arm == "curriculum" and index % 2 == 1
    stage = (
        "peaceful_agent"
        if index < 100
        else ("coin_collector_agent" if index < 500 else "rule_based_agent")
    )
    return {
        "kind": "hunting" if hunting else "classic",
        "opponents": [stage if hunting else "rule_based_agent"] * 3,
    }


class UpdateTracker:
    """Sidecar state keeps sampling provenance and cadence reproducible on resume."""

    def __init__(self, state, *, capacity, every, maximum_updates, origin):
        self.state = deepcopy(state)
        self.origins = deque(state["origins"], maxlen=capacity)
        self.every, self.maximum_updates, self.origin = every, maximum_updates, origin

    def record(self, policy, **transition):
        policy.replay_buffer.add(**transition)
        self.origins.append(self.origin)
        policy.episode_reward += transition["reward"]
        self.state["transitions"] += 1
        self.state["generated"][self.origin] = self.state["generated"].get(self.origin, 0) + 1
        if (
            self.state["transitions"] % self.every
            or policy.learner.update_steps >= self.maximum_updates
            or len(policy.replay_buffer) < policy.config.replay_warmup
        ):
            return
        # Clone the RNG to observe exactly the indices sampled by the original
        # uniform sampler, without advancing it or modifying the batch.
        observed_rng = np.random.default_rng()
        observed_rng.bit_generator.state = deepcopy(policy.replay_buffer._rng.bit_generator.state)
        indices = observed_rng.choice(len(self.origins), policy.config.batch_size, replace=False)
        counts = Counter(self.origins[int(i)] for i in indices)
        for key, value in counts.items():
            self.state["sampled"][key] = self.state["sampled"].get(key, 0) + value
        result = policy.learner.train_batch(policy.replay_buffer.sample(policy.config.batch_size))
        policy.losses.append(result.loss)
        policy.absolute_td_errors.append(result.mean_abs_td_error)
        if result.target_synchronized:
            policy.episode_target_synchronizations += 1

    def snapshot(self):
        return {**deepcopy(self.state), "origins": list(self.origins)}


class BombCredits:
    """Observe native bomb/explosion ownership without changing game events."""

    def __init__(self):
        self.bombs = {}
        self.explosions = {}

    def placed(self, bomb, *, step, safe_attack):
        self.bombs[bomb] = {
            "step": int(step),
            "safe_attack": bool(safe_attack),
            "detonated": False,
            "kill_credits": 0,
            "self_kill_credits": 0,
        }

    def detonated(self, bomb, explosion):
        if bomb in self.bombs:
            self.bombs[bomb]["detonated"] = True
            self.explosions[explosion] = self.bombs[bomb]

    def observe_hits(self, explosions, active_agents):
        # Mirrors native credit semantics, including simultaneous explosions.
        # These are credits, not a counterfactual claim that the placement was optimal.
        for explosion in explosions:
            if explosion not in self.explosions or not explosion.is_dangerous():
                continue
            for agent in active_agents:
                if not agent.dead and (agent.x, agent.y) in explosion.blast_coords:
                    key = "self_kill_credits" if agent is explosion.owner else "kill_credits"
                    self.explosions[explosion][key] += 1

    def snapshot(self):
        return [dict(record) for record in self.bombs.values()]
