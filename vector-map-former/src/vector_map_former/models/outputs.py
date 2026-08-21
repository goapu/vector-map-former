"""Stable model-output contract."""

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True, slots=True)
class ModelOutput:
    action_logits: Tensor
    movements: Tensor
