"""Training-only teacher loss and causally aligned replay annotations for #225."""

from collections import Counter, deque
from copy import deepcopy

import numpy as np
import torch
from torch.nn import functional as F

from training.hunting_curriculum import UpdateTracker


def distillation_loss(student, teacher, legal, retention, temperature=1.0):
    """Mean legal-action KL(teacher || student), excluding unknown/loop states."""
    if temperature <= 0 or student.shape != teacher.shape or legal.shape != student.shape:
        raise ValueError("Invalid teacher loss shape or temperature")
    if legal.dtype != torch.bool or retention.dtype != torch.bool:
        raise ValueError("Teacher masks must be boolean")
    if retention.shape != (student.shape[0],) or not bool(legal.any(dim=1).all()):
        raise ValueError("Invalid retention or legal-action mask")
    if not bool(torch.isfinite(student).all() and torch.isfinite(teacher).all()):
        raise ValueError("Non-finite teacher/student values")
    if not bool(retention.any()):
        return student.sum() * 0
    scaled = (student[retention] / temperature).masked_fill(~legal[retention], -1e9)
    fixed = (teacher.detach()[retention] / temperature).masked_fill(~legal[retention], -1e9)
    probabilities = F.softmax(fixed, dim=1)
    return (
        F.kl_div(F.log_softmax(scaled, dim=1), probabilities, reduction="batchmean")
        * temperature**2
    )


class TeacherTracker(UpdateTracker):
    """Keep annotations aligned with FIFO replay, including delayed terminal flushes."""

    def __init__(
        self,
        state,
        *,
        capacity,
        every,
        maximum_updates,
        origin,
        warmup,
        weight,
        teacher,
        temperature=1.0,
    ):
        super().__init__(
            state, capacity=capacity, every=every, maximum_updates=maximum_updates, origin=origin
        )
        if weight < 0 or warmup < 0:
            raise ValueError("Invalid teacher weight or warmup")
        metadata = state.get("teacher_metadata", [None] * len(self.origins))
        if len(metadata) != len(self.origins):
            raise ValueError("Replay annotation alignment lost")
        self.metadata = deque(deepcopy(metadata), maxlen=capacity)
        self.warmup, self.weight, self.temperature = warmup, weight, temperature
        self.teacher = teacher.eval().requires_grad_(False)
        self.observations = {}
        self.last_identity = None
        self.history = None
        self.state.setdefault("teacher_counts", {})
        self.state.setdefault("teacher_loss_sum", 0.0)
        self.state.setdefault("teacher_loss_updates", 0)

    def observe(self, state):
        from agent_code.DagobertDuckDQNAntiLoop.legality import framework_legal_action_mask
        from agent_code.DagobertDuckDQNAntiLoop.loop_reward import LoopReward

        if self.history is None:
            self.history = LoopReward()
        self.history.observe(state)
        window = list(self.history.window)
        is_loop = (
            len(window) == 24
            and len({r[0] for r in window}) <= 3
            and not any(r[-1] for r in window)
            and len({r[1:5] for r in window}) == 1
        )
        identity = (state["round"], state["step"])
        self.observations[identity] = {
            "loop": is_loop,
            "legal": framework_legal_action_mask(state).tolist(),
        }
        self.last_identity = identity

    def record(self, policy, **transition):
        pending = policy.pending_transition
        identity = pending.identity if pending is not None else self.last_identity
        if pending is not None and not np.array_equal(pending.state, transition["state"]):
            raise ValueError("Pending teacher annotation does not match replay state")
        if identity not in self.observations:
            raise ValueError("Missing causal teacher observation")
        metadata = self.observations.pop(identity)
        policy.replay_buffer.add(**transition)
        self.origins.append(self.origin)
        self.metadata.append(metadata)
        if len(self.metadata) != len(policy.replay_buffer):
            raise ValueError("Teacher annotations and replay differ in length")
        policy.episode_reward += transition["reward"]
        self.state["transitions"] += 1
        self.state["generated"][self.origin] = self.state["generated"].get(self.origin, 0) + 1
        counts = self.state["teacher_counts"]
        key = "generated_loop" if metadata["loop"] else "generated_retention"
        counts[key] = counts.get(key, 0) + 1
        if (
            self.state["transitions"] <= self.warmup
            or self.state["transitions"] % self.every
            or policy.learner.update_steps >= self.maximum_updates
            or len(policy.replay_buffer) < policy.config.replay_warmup
        ):
            return
        observed_rng = np.random.default_rng()
        observed_rng.bit_generator.state = deepcopy(policy.replay_buffer._rng.bit_generator.state)
        indices = observed_rng.choice(len(self.origins), policy.config.batch_size, replace=False)
        sampled = [self.metadata[int(i)] for i in indices]
        for key, value in Counter(self.origins[int(i)] for i in indices).items():
            self.state["sampled"][key] = self.state["sampled"].get(key, 0) + value
        for row in sampled:
            key = (
                "sampled_unknown"
                if row is None
                else ("sampled_loop" if row["loop"] else "sampled_retention")
            )
            counts[key] = counts.get(key, 0) + 1
        legal = torch.tensor([r["legal"] if r else [True] * 6 for r in sampled], dtype=torch.bool)
        retention = torch.tensor(
            [r is not None and not r["loop"] for r in sampled], dtype=torch.bool
        )

        def loss(states, values):
            with torch.no_grad():
                teacher_values = self.teacher(states)
            value = distillation_loss(values, teacher_values, legal, retention, self.temperature)
            self.state["teacher_loss_sum"] += float(value.detach())
            self.state["teacher_loss_updates"] += 1
            return self.weight * value

        policy.learner.auxiliary_loss = loss if self.weight else None
        try:
            result = policy.learner.train_batch(
                policy.replay_buffer.sample(policy.config.batch_size)
            )
        finally:
            policy.learner.auxiliary_loss = None
        policy.losses.append(result.loss)
        policy.absolute_td_errors.append(result.mean_abs_td_error)
        if result.target_synchronized:
            policy.episode_target_synchronizations += 1

    def snapshot(self):
        return {**super().snapshot(), "teacher_metadata": deepcopy(list(self.metadata))}


def exposure_gate(tracker, minimum):
    counts = tracker.get("teacher_counts", {})
    keys = ("generated_loop", "sampled_loop", "sampled_retention")
    return {key: counts.get(key, 0) >= minimum[key] for key in keys}
