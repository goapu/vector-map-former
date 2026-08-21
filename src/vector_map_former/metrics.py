"""Dependency-light streaming metrics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from vector_map_former.constants import NUM_ACTIONS, PAD_ACTION, Action


@dataclass(frozen=True, slots=True)
class ClassificationReport:
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    per_class: dict[str, dict[str, float]]
    confusion_matrix: list[list[int]]

    def to_dict(self) -> dict[str, object]:
        return {
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "macro_f1": self.macro_f1,
            "per_class": self.per_class,
            "confusion_matrix": self.confusion_matrix,
        }


class ClassificationAccumulator:
    def __init__(self) -> None:
        self.confusion = torch.zeros(NUM_ACTIONS, NUM_ACTIONS, dtype=torch.long)

    def update(self, logits: Tensor, targets: Tensor) -> None:
        predictions = logits.argmax(dim=-1).detach().cpu().reshape(-1)
        targets = targets.detach().cpu().reshape(-1)
        valid = targets != PAD_ACTION
        predictions = predictions[valid]
        targets = targets[valid]
        if not len(targets):
            return
        indices = targets * NUM_ACTIONS + predictions
        counts = torch.bincount(indices, minlength=NUM_ACTIONS**2)
        self.confusion += counts.reshape(NUM_ACTIONS, NUM_ACTIONS)

    def compute(self) -> ClassificationReport:
        matrix = self.confusion.to(dtype=torch.float64)
        total = matrix.sum().item()
        accuracy = float(matrix.diag().sum().item() / total) if total else 0.0
        per_class: dict[str, dict[str, float]] = {}
        recalls: list[float] = []
        f1_scores: list[float] = []
        for action in Action:
            index = int(action)
            true_positive = matrix[index, index].item()
            predicted = matrix[:, index].sum().item()
            actual = matrix[index, :].sum().item()
            precision = true_positive / predicted if predicted else 0.0
            recall = true_positive / actual if actual else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            recalls.append(recall)
            f1_scores.append(f1)
            per_class[action.name] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": int(actual),
            }
        return ClassificationReport(
            accuracy=accuracy,
            balanced_accuracy=float(sum(recalls) / len(recalls)),
            macro_f1=float(sum(f1_scores) / len(f1_scores)),
            per_class=per_class,
            confusion_matrix=self.confusion.tolist(),
        )


class MovementAccumulator:
    """Movement error evaluated only on true MOVE vertices."""

    def __init__(self) -> None:
        self.absolute_error = 0.0
        self.euclidean_error = 0.0
        self.values = 0
        self.vertices = 0

    def update(self, predictions: Tensor, targets: Tensor, actions: Tensor) -> None:
        mask = actions == int(Action.MOVE)
        if not mask.any():
            return
        difference = (predictions[mask] - targets[mask]).detach().cpu()
        self.absolute_error += float(difference.abs().sum().item())
        self.euclidean_error += float(torch.linalg.vector_norm(difference, dim=-1).sum().item())
        self.values += int(difference.numel())
        self.vertices += int(difference.shape[0])

    def compute(self) -> dict[str, float | int]:
        return {
            "component_mae_normalized": self.absolute_error / self.values if self.values else 0.0,
            "endpoint_error_normalized": (
                self.euclidean_error / self.vertices if self.vertices else 0.0
            ),
            "move_vertices": self.vertices,
        }
