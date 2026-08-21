"""Constraint-aware polygon reconstruction and optional geometry metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vector_map_former.constants import Action


def _moved_coordinate(
    coordinates: np.ndarray,
    index: int,
    edge_movements: np.ndarray,
    epsilon: float = 1e-10,
) -> np.ndarray:
    current = coordinates[index]
    previous = coordinates[index - 1]
    following = coordinates[(index + 1) % len(coordinates)]
    incoming = current - previous
    outgoing = following - current
    incoming_length = float(np.linalg.norm(incoming))
    outgoing_length = float(np.linalg.norm(outgoing))
    system = np.asarray([incoming, outgoing], dtype=np.float64)
    right_hand = np.asarray(
        [incoming_length * edge_movements[0], outgoing_length * edge_movements[1]],
        dtype=np.float64,
    )
    if abs(float(np.linalg.det(system))) <= epsilon:
        return np.asarray(current.copy(), dtype=np.float64)
    moved = current + np.linalg.solve(system, right_hand)
    return np.asarray(moved, dtype=np.float64)


def reconstruct_polygon(
    coordinates: np.ndarray,
    actions: np.ndarray,
    movements: np.ndarray,
) -> np.ndarray | None:
    """Reconstruct a ring while preserving input order; return None if under-defined."""

    coordinates = np.asarray(coordinates, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.int64)
    movements = np.asarray(movements, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must have shape [vertices, 2]")
    if actions.shape != (len(coordinates),) or movements.shape != (len(coordinates), 2):
        raise ValueError("actions or movements do not match coordinate length")
    output: list[np.ndarray] = []
    for index, action in enumerate(actions):
        if action == int(Action.REMOVE):
            continue
        if action == int(Action.MOVE):
            output.append(_moved_coordinate(coordinates, index, movements[index]))
        elif action == int(Action.KEEP):
            output.append(coordinates[index].copy())
        else:
            raise ValueError(f"invalid action {action}")
    if len(output) < 3:
        return None
    return np.asarray(output, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class GeometryReport:
    evaluated: int
    invalid_predictions: int
    invalid_targets: int
    valid_percentage: float
    mean_iou: float | None
    mean_hausdorff: float | None
    mean_relative_area_error: float | None

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "evaluated": self.evaluated,
            "invalid_predictions": self.invalid_predictions,
            "invalid_targets": self.invalid_targets,
            "valid_percentage": self.valid_percentage,
            "mean_iou": self.mean_iou,
            "mean_hausdorff": self.mean_hausdorff,
            "mean_relative_area_error": self.mean_relative_area_error,
        }


class GeometryAccumulator:
    """Shapely-backed polygon metrics kept separate from the training hot path."""

    def __init__(self) -> None:
        try:
            from shapely.geometry import Polygon  # noqa: F401
        except ImportError as error:
            raise RuntimeError(
                "Geometry metrics require the 'geometry' optional dependencies"
            ) from error
        self.evaluated = 0
        self.invalid_predictions = 0
        self.invalid_targets = 0
        self.ious: list[float] = []
        self.hausdorff: list[float] = []
        self.area_errors: list[float] = []

    def update(
        self,
        source_coordinates: np.ndarray,
        predicted_actions: np.ndarray,
        predicted_movements: np.ndarray,
        target_actions: np.ndarray,
        target_movements: np.ndarray,
    ) -> None:
        from shapely.geometry import Polygon

        self.evaluated += 1
        predicted_ring = reconstruct_polygon(
            source_coordinates,
            predicted_actions,
            predicted_movements,
        )
        target_ring = reconstruct_polygon(
            source_coordinates,
            target_actions,
            target_movements,
        )
        if target_ring is None:
            self.invalid_targets += 1
            return
        target = Polygon(target_ring)
        if not target.is_valid or target.is_empty:
            self.invalid_targets += 1
            return
        if predicted_ring is None:
            self.invalid_predictions += 1
            return
        predicted = Polygon(predicted_ring)
        if not predicted.is_valid or predicted.is_empty:
            self.invalid_predictions += 1
            return
        union = predicted.union(target).area
        self.ious.append(float(predicted.intersection(target).area / union) if union else 0.0)
        self.hausdorff.append(float(predicted.hausdorff_distance(target)))
        target_area = float(target.area)
        self.area_errors.append(
            abs(float(predicted.area) - target_area) / target_area if target_area else 0.0
        )

    def compute(self) -> GeometryReport:
        eligible = self.evaluated - self.invalid_targets
        valid = eligible - self.invalid_predictions
        return GeometryReport(
            evaluated=self.evaluated,
            invalid_predictions=self.invalid_predictions,
            invalid_targets=self.invalid_targets,
            valid_percentage=100.0 * valid / eligible if eligible else 0.0,
            mean_iou=float(np.mean(self.ious)) if self.ious else None,
            mean_hausdorff=float(np.mean(self.hausdorff)) if self.hausdorff else None,
            mean_relative_area_error=(
                float(np.mean(self.area_errors)) if self.area_errors else None
            ),
        )
