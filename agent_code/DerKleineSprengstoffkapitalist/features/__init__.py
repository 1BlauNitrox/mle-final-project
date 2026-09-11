"""Stable public feature API for the tabular successor."""

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
from .representations import (
    BASELINE_STATE_REPRESENTATION,
    COMPACT_STATE_REPRESENTATION,
    VALID_STATE_REPRESENTATIONS,
    StateRepresentation,
    encode_state,
    get_state_representation,
)

__all__ = [
    "FEATURE_COUNT",
    "FEATURE_DOMAINS",
    "FEATURE_NAMES",
    "FEATURE_SCHEMA_VERSION",
    "THEORETICAL_STATE_SPACE_UPPER_BOUND",
    "StateFeatures",
    "state_to_features",
    "validate_features",
    "BASELINE_STATE_REPRESENTATION",
    "COMPACT_STATE_REPRESENTATION",
    "VALID_STATE_REPRESENTATIONS",
    "StateRepresentation",
    "encode_state",
    "get_state_representation",
]
