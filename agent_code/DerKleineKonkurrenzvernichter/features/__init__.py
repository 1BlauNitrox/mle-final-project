"""Public feature API for the opponent-aware tabular agent."""

from .assemble import (
    FEATURE_COUNT,
    FEATURE_DOMAINS,
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    THEORETICAL_STATE_SPACE_UPPER_BOUND,
    StateFeatures,
    state_to_features,
    validate_features,
)
from .opponents import OPPONENT_STATE_REPRESENTATION
from .representations import (
    BASELINE_STATE_REPRESENTATION,
    COMPACT_ESCAPE_STATE_REPRESENTATION,
    COMPACT_STATE_REPRESENTATION,
    VALID_STATE_REPRESENTATIONS,
    StateRepresentation,
    encode_state,
    get_state_representation,
)
from .shared_targets import SHARED_TARGET_STATE_REPRESENTATION

__all__ = [
    "FEATURE_COUNT",
    "FEATURE_DOMAINS",
    "FEATURE_NAMES",
    "FEATURE_SCHEMA_VERSION",
    "THEORETICAL_STATE_SPACE_UPPER_BOUND",
    "StateFeatures",
    "OPPONENT_STATE_REPRESENTATION",
    "SHARED_TARGET_STATE_REPRESENTATION",
    "state_to_features",
    "validate_features",
    "BASELINE_STATE_REPRESENTATION",
    "COMPACT_STATE_REPRESENTATION",
    "COMPACT_ESCAPE_STATE_REPRESENTATION",
    "VALID_STATE_REPRESENTATIONS",
    "StateRepresentation",
    "encode_state",
    "get_state_representation",
]
