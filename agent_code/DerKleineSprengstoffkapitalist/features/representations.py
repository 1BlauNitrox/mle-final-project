"""State-representation selection for the tabular Task 2 agent."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .assemble import (
    FEATURE_COUNT,
    FEATURE_SCHEMA_VERSION,
    StateFeatures,
    state_to_features,
    validate_features,
)
from .compact import (
    COMPACT_FEATURE_COUNT,
    COMPACT_FEATURE_SCHEMA_VERSION,
    compact_state_to_features,
    validate_compact_features,
)
from .compact_escape import (
    COMPACT_ESCAPE_FEATURE_COUNT,
    COMPACT_ESCAPE_FEATURE_SCHEMA_VERSION,
    compact_escape_state_to_features,
    validate_compact_escape_features,
)

BASELINE_STATE_REPRESENTATION = "baseline"
COMPACT_STATE_REPRESENTATION = "compact_decision"
COMPACT_ESCAPE_STATE_REPRESENTATION = "compact_post_bomb_escape"

VALID_STATE_REPRESENTATIONS = (
    BASELINE_STATE_REPRESENTATION,
    COMPACT_STATE_REPRESENTATION,
    COMPACT_ESCAPE_STATE_REPRESENTATION,
)


@dataclass(frozen=True)
class StateRepresentation:
    """Runtime contract for one state-feature schema."""

    name: str
    feature_count: int
    feature_schema_version: int
    encode: Callable[[dict | None], StateFeatures | None]
    validate: Callable[[StateFeatures], None]


REPRESENTATIONS = {
    BASELINE_STATE_REPRESENTATION: StateRepresentation(
        name=BASELINE_STATE_REPRESENTATION,
        feature_count=FEATURE_COUNT,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        encode=state_to_features,
        validate=validate_features,
    ),
    COMPACT_STATE_REPRESENTATION: StateRepresentation(
        name=COMPACT_STATE_REPRESENTATION,
        feature_count=COMPACT_FEATURE_COUNT,
        feature_schema_version=COMPACT_FEATURE_SCHEMA_VERSION,
        encode=compact_state_to_features,
        validate=validate_compact_features,
    ),
    COMPACT_ESCAPE_STATE_REPRESENTATION: StateRepresentation(
        name=COMPACT_ESCAPE_STATE_REPRESENTATION,
        feature_count=COMPACT_ESCAPE_FEATURE_COUNT,
        feature_schema_version=COMPACT_ESCAPE_FEATURE_SCHEMA_VERSION,
        encode=compact_escape_state_to_features,
        validate=validate_compact_escape_features,
    ),
}


def get_state_representation(name: str) -> StateRepresentation:
    """Return the validated state-representation contract."""

    try:
        return REPRESENTATIONS[name]
    except KeyError as error:
        raise ValueError(
            f"State representation must be one of "
            f"{list(VALID_STATE_REPRESENTATIONS)}."
        ) from error


def encode_state(
    game_state: dict | None,
    representation: str = BASELINE_STATE_REPRESENTATION,
) -> StateFeatures | None:
    """Encode a game state using the selected representation."""

    return get_state_representation(representation).encode(game_state)
