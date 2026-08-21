"""Validated adapter for the public MapGeneralizer NumPy arrays."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from vector_map_former.constants import NUM_ACTIONS, Action
from vector_map_former.data.contracts import PolygonSample
from vector_map_former.data.features import build_features, normalize_polygon

Split = Literal["train", "valid", "test"]
_ALLOWED_SPLITS: tuple[Split, ...] = ("train", "valid", "test")
_EXPECTED_COLUMNS = 13


class MapGeneralizerDataset(Dataset[PolygonSample]):
    """One object-array element per building, with strict schema validation."""

    def __init__(
        self,
        data_dir: str | Path,
        split: Split,
        *,
        feature_set: str = "xy",
        normalize_movement: bool = True,
        validate_on_load: bool = True,
        max_vertices: int | None = None,
    ) -> None:
        if split not in _ALLOWED_SPLITS:
            raise ValueError(f"split must be one of {_ALLOWED_SPLITS}")
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.split = split
        self.feature_set = feature_set
        self.normalize_movement = normalize_movement
        self.max_vertices = max_vertices
        self.path = self.data_dir / f"vertex_{split}.npy"
        if not self.path.is_file():
            raise FileNotFoundError(f"MapGeneralizer split not found: {self.path}")

        # MapGeneralizer distributes object arrays. Only load the exact expected
        # filenames from a user-selected directory; never enable pickle for an
        # arbitrary input filename.
        loaded = np.load(self.path, allow_pickle=True)
        if not isinstance(loaded, np.ndarray) or loaded.ndim != 1:
            raise ValueError(f"{self.path.name} must be a 1D NumPy object array")
        if loaded.size == 0:
            raise ValueError(f"{self.path.name} must contain at least one building")
        self._polygons = loaded
        if validate_on_load:
            self._validate_collection()

    def __len__(self) -> int:
        return len(self._polygons)

    def __getitem__(self, index: int) -> PolygonSample:
        matrix = self._matrix(index)
        order = matrix[:, 1].astype(np.int64)
        sort_index = np.argsort(order, kind="stable")
        matrix = matrix[sort_index]

        raw_coordinates = matrix[:, 2:4]
        coordinates, centroid, scale = normalize_polygon(raw_coordinates)
        movements = matrix[:, 11:13].astype(np.float32)
        if self.normalize_movement:
            movements = movements / np.float32(scale)
        actions = matrix[:, 10].astype(np.int64)
        features = build_features(coordinates, self.feature_set)

        sample = PolygonSample(
            building_id=int(matrix[0, 0]),
            features=torch.from_numpy(features),
            coordinates=torch.from_numpy(coordinates),
            raw_coordinates=torch.from_numpy(raw_coordinates.astype(np.float32)),
            actions=torch.from_numpy(actions),
            movements=torch.from_numpy(movements),
            centroid=torch.from_numpy(centroid),
            scale=torch.tensor(scale, dtype=torch.float32),
        )
        sample.validate()
        return sample

    def action_counts(self) -> torch.Tensor:
        """Return counts in REMOVE, KEEP, MOVE order."""

        counts = np.zeros(NUM_ACTIONS, dtype=np.int64)
        for index in range(len(self)):
            actions = self._matrix(index)[:, 10].astype(np.int64)
            counts += np.bincount(actions, minlength=NUM_ACTIONS)
        return torch.from_numpy(counts)

    def summary(self) -> dict[str, object]:
        """Return a JSON-serializable audit summary."""

        lengths = np.asarray([self._matrix(i).shape[0] for i in range(len(self))])
        action_counts = self.action_counts().tolist()
        return {
            "split": self.split,
            "path": str(self.path),
            "buildings": len(self),
            "vertices": int(lengths.sum()),
            "vertices_per_building": {
                "min": int(lengths.min()),
                "median": float(np.median(lengths)),
                "p95": float(np.percentile(lengths, 95)),
                "max": int(lengths.max()),
            },
            "action_counts": {
                Action.REMOVE.name: action_counts[Action.REMOVE],
                Action.KEEP.name: action_counts[Action.KEEP],
                Action.MOVE.name: action_counts[Action.MOVE],
            },
        }

    def building_ids(self) -> set[int]:
        return {int(self._matrix(i)[0, 0]) for i in range(len(self))}

    def _matrix(self, index: int) -> np.ndarray:
        matrix = np.asarray(self._polygons[index], dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != _EXPECTED_COLUMNS:
            raise ValueError(
                f"building at index {index} has shape {matrix.shape}; "
                f"expected [vertices, {_EXPECTED_COLUMNS}]"
            )
        return matrix

    def _validate_collection(self) -> None:
        seen_ids: set[int] = set()
        failures: Counter[str] = Counter()
        for index in range(len(self)):
            try:
                matrix = self._matrix(index)
                self._validate_matrix(matrix, index, seen_ids)
            except ValueError as error:
                failures[str(error)] += 1
                if sum(failures.values()) >= 10:
                    break
        if failures:
            details = "; ".join(f"{message} ({count})" for message, count in failures.items())
            raise ValueError(f"MapGeneralizer validation failed: {details}")

    def _validate_matrix(self, matrix: np.ndarray, index: int, seen_ids: set[int]) -> None:
        length = matrix.shape[0]
        if length < 3:
            raise ValueError(f"building {index} has fewer than 3 vertices")
        if self.max_vertices is not None and length > self.max_vertices:
            raise ValueError(
                f"building {index} has {length} vertices, "
                f"exceeding max_vertices={self.max_vertices}"
            )
        if not np.isfinite(matrix).all():
            raise ValueError(f"building {index} contains non-finite values")
        raw_building_ids = matrix[:, 0]
        if not np.equal(raw_building_ids, np.rint(raw_building_ids)).all():
            raise ValueError(f"building {index} contains a non-integer building ID")
        building_ids = np.unique(raw_building_ids.astype(np.int64))
        if len(building_ids) != 1:
            raise ValueError(f"building {index} contains multiple building IDs")
        building_id = int(building_ids[0])
        if building_id in seen_ids:
            raise ValueError(f"duplicate building ID {building_id}")
        seen_ids.add(building_id)

        raw_order = matrix[:, 1]
        if not np.equal(raw_order, np.rint(raw_order)).all():
            raise ValueError(f"building {building_id} has non-integer vertex order")
        order = raw_order.astype(np.int64)
        if not np.array_equal(np.sort(order), np.arange(length)):
            raise ValueError(f"building {building_id} has invalid vertex order")
        raw_actions = matrix[:, 10]
        if not np.equal(raw_actions, np.rint(raw_actions)).all():
            raise ValueError(f"building {building_id} has non-integer action labels")
        actions = raw_actions.astype(np.int64)
        if not np.isin(actions, [int(action) for action in Action]).all():
            raise ValueError(f"building {building_id} has invalid action labels")
