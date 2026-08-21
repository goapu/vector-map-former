"""Masked multi-task losses for action classification and vertex movement."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from vector_map_former.constants import PAD_ACTION, Action
from vector_map_former.data.collate import PolygonBatch
from vector_map_former.models.outputs import ModelOutput


@dataclass(frozen=True, slots=True)
class LossOutput:
    total: Tensor
    classification: Tensor
    movement: Tensor
    valid_vertices: int
    moved_vertices: int


class MultiTaskCriterion(nn.Module):
    """Action loss plus movement regression restricted to true MOVE vertices."""

    class_weights: Tensor

    def __init__(
        self,
        *,
        class_weights: Tensor | None,
        movement_weight: float,
    ) -> None:
        super().__init__()
        if movement_weight < 0:
            raise ValueError("movement_weight must be non-negative")
        self.movement_weight = movement_weight
        weights = torch.empty(0) if class_weights is None else class_weights.float()
        self.register_buffer("class_weights", weights)

    def forward(self, output: ModelOutput, batch: PolygonBatch) -> LossOutput:
        weights = self.class_weights if self.class_weights.numel() else None
        classification = functional.cross_entropy(
            output.action_logits.transpose(1, 2),
            batch.actions,
            weight=weights,
            ignore_index=PAD_ACTION,
        )
        move_mask = (batch.actions == int(Action.MOVE)) & ~batch.padding_mask
        moved_vertices = int(move_mask.sum().item())
        if moved_vertices:
            movement = functional.smooth_l1_loss(
                output.movements[move_mask],
                batch.movements[move_mask],
            )
        else:
            movement = output.movements.sum() * 0.0
        total = classification + self.movement_weight * movement
        return LossOutput(
            total=total,
            classification=classification.detach(),
            movement=movement.detach(),
            valid_vertices=int((~batch.padding_mask).sum().item()),
            moved_vertices=moved_vertices,
        )


def balanced_class_weights(counts: Tensor) -> Tensor:
    """Compute inverse-frequency weights with mean weight close to one."""

    counts = counts.to(dtype=torch.float64)
    if counts.ndim != 1 or torch.any(counts <= 0):
        raise ValueError("class counts must be a positive one-dimensional tensor")
    weights = counts.sum() / (len(counts) * counts)
    return weights.to(dtype=torch.float32)
