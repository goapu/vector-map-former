from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _building(building_id: int, vertices: int, offset: float = 0.0) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, vertices, endpoint=False)
    coordinates = np.stack([np.cos(angles), np.sin(angles)], axis=1) * 10.0 + offset
    matrix = np.zeros((vertices, 13), dtype=np.float64)
    matrix[:, 0] = building_id
    matrix[:, 1] = np.arange(vertices)
    matrix[:, 2:4] = coordinates
    matrix[:, 10] = np.arange(vertices) % 3
    matrix[:, 11:13] = 0.05
    return matrix


@pytest.fixture()
def tiny_data_dir(tmp_path: Path) -> Path:
    identifiers = {
        "train": (100, 101, 102, 103, 104, 105),
        "valid": (200, 201, 202),
        "test": (300, 301, 302),
    }
    for split, building_ids in identifiers.items():
        values = np.asarray(
            [
                _building(building_id, vertices=4 + index % 3, offset=float(index) * 20.0)
                for index, building_id in enumerate(building_ids)
            ],
            dtype=object,
        )
        np.save(tmp_path / f"vertex_{split}.npy", values)
    return tmp_path
