"""Training-only interventions for the registered Task 3 approach pilots.

Evaluation always loads the unchanged archived agent: nothing here is imported
or installed when `training` is false, and no evaluation-time action rule is
introduced. Every knob is restricted to its registered values so an
unregistered configuration fails before any episode is played.

The approach potential is a pure function of the public game state, so
`F(s, s') = gamma * Phi(s') - Phi(s)` is potential-based shaping in the sense of
Ng, Harada and Russell (1999) and leaves the optimal policy unchanged. The
telescoping sum of `F` over a trajectory is `gamma^T * Phi(s_T) - Phi(s_0)`,
so every bonus collected while approaching an opponent is paid back exactly
when the opponent is eliminated and `Phi` returns to zero. The shaping
redistributes existing credit earlier in time; it does not add return.
"""

from __future__ import annotations

from contextlib import contextmanager

from scripts.frozen_opponent_inputs import FrozenOpponentInputs

REGISTERED_POTENTIAL_SCALES = (0.0, 1.0, 3.0)
REGISTERED_UPDATE_INTERVALS = (1, 8)
# A bomb reaches three tiles and fuses in four steps, so an opponent beyond
# roughly eight tiles cannot be converted into a kill within the horizon the
# agent is deciding over. `None` admits every transition.
REGISTERED_PROXIMITY_GATES = (None, 8)
REGISTERED_L2_COEFFICIENTS = (0.0, 0.1, 1.0)
FIRST_LAYER = "layers.0.weight"
OPPONENT_PREFIX = 26


def nearest_opponent_distance(game_state: dict | None) -> int | None:
    """Return the Manhattan distance to the nearest public opponent.

    This reuses the metric and the public-position restriction of the
    registered opponent inputs (contract indices 27-29) and of the inherited
    `MOVED_TOWARDS_COIN` shaping, rather than introducing a second notion of
    distance. `None` means no opponent is currently visible.
    """
    if game_state is None:
        return None

    others = game_state.get("others") or []
    if not others:
        return None

    x, y = game_state["self"][3]
    # Framework positions can be NumPy integers; return a plain int so callers
    # can record the value in JSON evidence without a conversion step.
    return int(min(abs(other[3][0] - x) + abs(other[3][1] - y) for other in others))


def approach_potential(game_state: dict | None, *, scale: float, discount_factor: float) -> float:
    """Return the registered opponent-approach potential `scale * gamma ** d`.

    This is the discounted value of a `scale`-sized reward `d` steps away, that
    is, the value function of a hypothetical agent that walks straight at the
    nearest opponent. Using it as `Phi` makes the shaping exactly cancel the
    discounting along an approach, which a potential linear in distance does
    not: at `gamma = 0.9` a board-wide linear potential has a drift term
    `(1 - gamma) * Phi` several times larger than its own per-step gradient, so
    it would penalise closing distance at every realistic separation.

    With this form the shaped term is exactly `0` for a step that reduces the
    distance by one, `-(1 - gamma) * Phi` for standing still, and
    `-(1 - gamma**2) * Phi` for retreating. `Phi` is zero without a visible
    opponent, and decays to zero with distance, so distant states are
    essentially unshaped.
    """
    if scale not in REGISTERED_POTENTIAL_SCALES:
        raise ValueError("Unregistered approach-potential scale")
    if not 0.0 < discount_factor <= 1.0:
        raise ValueError("Discount factor must be in (0, 1].")

    distance = nearest_opponent_distance(game_state)
    if distance is None:
        return 0.0

    return scale * discount_factor**distance


def potential_reward(
    current_potential: float,
    next_potential: float | None,
    *,
    terminal: bool,
    discount_factor: float,
) -> float:
    """Return `gamma * Phi(next) - Phi(current)`, with `Phi(terminal) = 0`."""
    if not 0.0 <= discount_factor <= 1.0:
        raise ValueError("Discount factor must be in [0, 1].")

    if terminal:
        if next_potential is not None:
            raise ValueError("Terminal shaping transitions cannot have a next potential.")
        successor = 0.0
    else:
        if next_potential is None:
            raise ValueError("Non-terminal shaping transitions require a next potential.")
        successor = float(next_potential)

    return discount_factor * successor - float(current_potential)


class RegularizedOpponentInputs(FrozenOpponentInputs):
    """Adds the gradient of `lambda/2 * sum(W_opponent**2)` before clipping.

    Shrinking only the trainable opponent columns toward zero shrinks the block
    toward the unchanged reference, because a zero opponent block reproduces the
    reference exactly. The coefficient is therefore a direct dial between "train
    freely" and "do not train at all", which is what makes a dose-response over
    it able to separate genuine opponent signal from undirected drift.
    """

    def __init__(self, learner, anchor, coefficient=0.0):
        if coefficient not in REGISTERED_L2_COEFFICIENTS:
            raise ValueError("Unregistered opponent L2 coefficient")
        self.coefficient = coefficient
        super().__init__(learner, anchor)

    def mask_gradient(self, gradient):
        result = FrozenOpponentInputs.mask_gradient(gradient)
        if self.coefficient:
            weights = self.parameters[FIRST_LAYER].detach()[:, OPPONENT_PREFIX:]
            result[:, OPPONENT_PREFIX:] += self.coefficient * weights
        return result


@contextmanager
def training_intervention(
    train,
    *,
    potential_scale,
    update_every,
    discount_factor,
    proximity_gate=None,
):
    """Install the registered training-only interventions on `train`.

    The replacement for `_record_transition` mirrors the archived runtime's own
    body, with three registered additions.

    The shaping term is added to the stored reward. Post-warmup transitions are
    counted so that only every `update_every`-th one samples a batch; skipped
    transitions still enter replay with their full reward, and target
    synchronization is still counted from actual optimizer updates.

    A proximity gate restricts what enters replay to transitions whose start
    state has an opponent within `proximity_gate` tiles. The update schedule
    still counts *every* post-warmup transition, so both gate settings perform
    essentially the same number of optimizer steps and the comparison isolates
    which experience supplies the gradient rather than how much gradient is
    applied.

    The opponent L2 coefficient is installed separately by the caller through
    `RegularizedOpponentInputs`, because it belongs to the optimizer guard
    rather than to the transition recorder.
    """
    if potential_scale not in REGISTERED_POTENTIAL_SCALES:
        raise ValueError("Unregistered approach-potential scale")
    if update_every not in REGISTERED_UPDATE_INTERVALS:
        raise ValueError("Unregistered update interval")
    if proximity_gate not in REGISTERED_PROXIMITY_GATES:
        raise ValueError("Unregistered proximity gate")

    original_events = train.game_events_occurred
    original_end_of_round = train.end_of_round

    def shaping_term(self, terminal):
        if not potential_scale:
            return 0.0
        if terminal:
            return potential_reward(
                getattr(self, "approach_terminal_potential", 0.0),
                None,
                terminal=True,
                discount_factor=discount_factor,
            )
        potentials = getattr(self, "approach_potentials", None)
        if potentials is None:
            return 0.0
        return potential_reward(
            potentials[0], potentials[1], terminal=False, discount_factor=discount_factor
        )

    def admitted(self, terminal):
        """Whether this transition's start state lies inside the proximity gate."""
        if proximity_gate is None:
            return True
        if terminal:
            distance = getattr(self, "approach_terminal_distance", None)
        else:
            pair = getattr(self, "approach_distances", None)
            distance = pair[0] if pair else None
        return distance is not None and distance <= proximity_gate

    def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
        # The original finalizes the *previous* pending transition first, and
        # that transition still needs the previous step's values. So this step's
        # values are computed first but only published afterwards.
        current = (
            approach_potential(
                old_game_state, scale=potential_scale, discount_factor=discount_factor
            ),
            approach_potential(
                new_game_state, scale=potential_scale, discount_factor=discount_factor
            ),
        )
        distances = (
            nearest_opponent_distance(old_game_state),
            nearest_opponent_distance(new_game_state),
        )
        original_events(self, old_game_state, self_action, new_game_state, events)
        pending = self.pending_transition is not None
        self.approach_potentials = current if pending else None
        self.approach_distances = distances if pending else None

    def end_of_round(self, last_game_state, last_action, events):
        # Both terminal paths record a transition that starts in this state:
        # the matching branch reuses the pending transition, whose identity the
        # runtime has already checked against `last_game_state`.
        self.approach_terminal_potential = approach_potential(
            last_game_state, scale=potential_scale, discount_factor=discount_factor
        )
        self.approach_terminal_distance = nearest_opponent_distance(last_game_state)
        try:
            return original_end_of_round(self, last_game_state, last_action, events)
        finally:
            self.approach_potentials = None
            self.approach_distances = None
            self.approach_terminal_potential = 0.0
            self.approach_terminal_distance = None

    def record_transition(
        self, *, state, action_index, reward, next_state, next_action_mask=None, terminal
    ):
        shaped = reward + shaping_term(self, terminal)
        self.episode_reward += shaped
        if admitted(self, terminal):
            self.replay_buffer.add(
                state=state,
                action_index=action_index,
                reward=shaped,
                next_state=next_state,
                terminal=terminal,
                next_action_mask=next_action_mask,
            )
        else:
            self.approach_gated_out = getattr(self, "approach_gated_out", 0) + 1

        if len(self.replay_buffer) < self.config.replay_warmup:
            return

        self.approach_eligible = getattr(self, "approach_eligible", 0) + 1
        if self.approach_eligible % update_every:
            return

        result = self.learner.train_batch(self.replay_buffer.sample(self.config.batch_size))
        self.losses.append(result.loss)
        self.absolute_td_errors.append(result.mean_abs_td_error)
        if result.target_synchronized:
            self.episode_target_synchronizations += 1

    previous = (original_events, original_end_of_round, train._record_transition)
    train.game_events_occurred = game_events_occurred
    train.end_of_round = end_of_round
    train._record_transition = record_transition
    try:
        yield
    finally:
        (
            train.game_events_occurred,
            train.end_of_round,
            train._record_transition,
        ) = previous
