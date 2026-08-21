"""Typed data contracts shared by adapters and models."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class PolygonSample:
    """One ordered exterior ring and its vertex-level targets."""

    building_id: int
    features: Tensor
    coordinates: Tensor
    raw_coordinates: Tensor
    actions: Tensor
    movements: Tensor
    centroid: Tensor
    scale: Tensor

    def validate(self) -> None:
        length = self.coordinates.shape[0]
        expected = {
            "coordinates": (length, 2),
            "raw_coordinates": (length, 2),
            "actions": (length,),
            "movements": (length, 2),
            "centroid": (2,),
            "scale": (),
        }
        actual = {
            "coordinates": tuple(self.coordinates.shape),
            "raw_coordinates": tuple(self.raw_coordinates.shape),
            "actions": tuple(self.actions.shape),
            "movements": tuple(self.movements.shape),
            "centroid": tuple(self.centroid.shape),
            "scale": tuple(self.scale.shape),
        }
        for name, shape in expected.items():
            if actual[name] != shape:
                raise ValueError(f"{name} has shape {actual[name]}, expected {shape}")
        if self.features.ndim != 2 or self.features.shape[0] != length:
            raise ValueError("features must have shape [vertices, channels]")
        if length < 3:
            raise ValueError("a polygon ring requires at least three unique vertices")
        for name in ("features", "coordinates", "raw_coordinates", "movements"):
            value = getattr(self, name)
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values")
        if self.scale.item() <= 0:
            raise ValueError("normalization scale must be positive")
