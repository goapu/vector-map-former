"""Shared domain constants."""

from enum import IntEnum


class Action(IntEnum):
    """Per-vertex generalization action encoded by MapGeneralizer."""

    REMOVE = 0
    KEEP = 1
    MOVE = 2


NUM_ACTIONS = len(Action)
PAD_ACTION = -100
FEATURE_SETS = frozenset({"xy", "xy_structural"})
MODEL_NAMES = frozenset({"mlp", "circular_cnn", "transformer", "ring_transformer"})
