"""Coordinate normalization and structural feature extraction."""

from __future__ import annotations

import numpy as np

from vector_map_former.constants import FEATURE_SETS


def polygon_centroid(coordinates: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    """Return an area-weighted polygon centroid with a degenerate fallback."""

    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must have shape [vertices, 2]")
    following = np.roll(coordinates, -1, axis=0)
    cross = coordinates[:, 0] * following[:, 1] - following[:, 0] * coordinates[:, 1]
    twice_area = float(cross.sum())
    if abs(twice_area) <= epsilon:
        return np.asarray(coordinates.mean(axis=0), dtype=np.float64)
    factor = 1.0 / (3.0 * twice_area)
    centroid_x = ((coordinates[:, 0] + following[:, 0]) * cross).sum() * factor
    centroid_y = ((coordinates[:, 1] + following[:, 1]) * cross).sum() * factor
    centroid = np.asarray([centroid_x, centroid_y], dtype=np.float64)
    return centroid


def normalize_polygon(
    coordinates: np.ndarray,
    epsilon: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Center a polygon and scale by its maximum bounding-box extent."""

    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must have shape [vertices, 2]")
    if coordinates.shape[0] < 3:
        raise ValueError("at least three vertices are required")
    if not np.isfinite(coordinates).all():
        raise ValueError("coordinates contain non-finite values")
    centroid = polygon_centroid(coordinates)
    centered = coordinates - centroid
    extent = np.ptp(coordinates, axis=0)
    scale = max(float(extent.max()), epsilon)
    return (centered / scale).astype(np.float32), centroid.astype(np.float32), scale


def structural_features(coordinates: np.ndarray) -> np.ndarray:
    """Compute cyclic local features from normalized ordered coordinates."""

    coordinates = np.asarray(coordinates, dtype=np.float32)
    previous = np.roll(coordinates, 1, axis=0)
    following = np.roll(coordinates, -1, axis=0)
    incoming = coordinates - previous
    outgoing = following - coordinates
    incoming_length = np.linalg.norm(incoming, axis=1, keepdims=True)
    outgoing_length = np.linalg.norm(outgoing, axis=1, keepdims=True)
    safe_incoming = np.maximum(incoming_length, 1e-8)
    safe_outgoing = np.maximum(outgoing_length, 1e-8)
    incoming_unit = incoming / safe_incoming
    outgoing_unit = outgoing / safe_outgoing
    cosine = np.clip((incoming_unit * outgoing_unit).sum(axis=1, keepdims=True), -1.0, 1.0)
    cross = (
        incoming_unit[:, 0] * outgoing_unit[:, 1]
        - incoming_unit[:, 1] * outgoing_unit[:, 0]
    )[:, None]
    sine = cross
    convexity = np.sign(cross)
    return np.concatenate(
        [
            coordinates,
            incoming,
            outgoing,
            incoming_length,
            outgoing_length,
            sine,
            cosine,
            convexity,
        ],
        axis=1,
        dtype=np.float32,
    )


def build_features(coordinates: np.ndarray, feature_set: str) -> np.ndarray:
    """Build the configured model inputs from normalized coordinates."""

    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature_set: {feature_set}")
    if feature_set == "xy":
        return np.asarray(coordinates, dtype=np.float32)
    return structural_features(coordinates)


def feature_dimension(feature_set: str) -> int:
    """Return the number of channels emitted by a feature set."""

    if feature_set == "xy":
        return 2
    if feature_set == "xy_structural":
        return 11
    raise ValueError(f"unknown feature_set: {feature_set}")
