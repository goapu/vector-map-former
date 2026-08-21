"""Deterministic variable-length polygon batching."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from vector_map_former.constants import PAD_ACTION
from vector_map_former.data.contracts import PolygonSample


@dataclass(frozen=True, slots=True)
class PolygonBatch:
    features: Tensor
    coordinates: Tensor
    raw_coordinates: Tensor
    actions: Tensor
    movements: Tensor
    padding_mask: Tensor
    lengths: Tensor
    building_ids: Tensor
    centroids: Tensor
    scales: Tensor

    def to(self, device: torch.device | str) -> PolygonBatch:
        return PolygonBatch(
            features=self.features.to(device),
            coordinates=self.coordinates.to(device),
            raw_coordinates=self.raw_coordinates.to(device),
            actions=self.actions.to(device),
            movements=self.movements.to(device),
            padding_mask=self.padding_mask.to(device),
            lengths=self.lengths.to(device),
            building_ids=self.building_ids.to(device),
            centroids=self.centroids.to(device),
            scales=self.scales.to(device),
        )


def make_collate_fn(max_vertices: int) -> Callable[[Sequence[PolygonSample]], PolygonBatch]:
    if max_vertices < 3:
        raise ValueError("max_vertices must be at least 3")

    def collate(samples: Sequence[PolygonSample]) -> PolygonBatch:
        if not samples:
            raise ValueError("cannot collate an empty batch")
        feature_dim = samples[0].features.shape[1]
        batch_size = len(samples)

        features = torch.zeros(batch_size, max_vertices, feature_dim, dtype=torch.float32)
        coordinates = torch.zeros(batch_size, max_vertices, 2, dtype=torch.float32)
        raw_coordinates = torch.zeros(batch_size, max_vertices, 2, dtype=torch.float32)
        actions = torch.full((batch_size, max_vertices), PAD_ACTION, dtype=torch.long)
        movements = torch.zeros(batch_size, max_vertices, 2, dtype=torch.float32)
        padding_mask = torch.ones(batch_size, max_vertices, dtype=torch.bool)
        lengths = torch.empty(batch_size, dtype=torch.long)
        building_ids = torch.empty(batch_size, dtype=torch.long)
        centroids = torch.empty(batch_size, 2, dtype=torch.float32)
        scales = torch.empty(batch_size, dtype=torch.float32)

        for batch_index, sample in enumerate(samples):
            sample.validate()
            length = sample.coordinates.shape[0]
            if length > max_vertices:
                raise ValueError(
                    f"building {sample.building_id} has {length} vertices; "
                    f"max_vertices={max_vertices}. Truncation is intentionally disabled."
                )
            if sample.features.shape[1] != feature_dim:
                raise ValueError("all samples in a batch must have the same feature dimension")
            region = slice(0, length)
            features[batch_index, region] = sample.features
            coordinates[batch_index, region] = sample.coordinates
            raw_coordinates[batch_index, region] = sample.raw_coordinates
            actions[batch_index, region] = sample.actions
            movements[batch_index, region] = sample.movements
            padding_mask[batch_index, region] = False
            lengths[batch_index] = length
            building_ids[batch_index] = sample.building_id
            centroids[batch_index] = sample.centroid
            scales[batch_index] = sample.scale

        return PolygonBatch(
            features=features,
            coordinates=coordinates,
            raw_coordinates=raw_coordinates,
            actions=actions,
            movements=movements,
            padding_mask=padding_mask,
            lengths=lengths,
            building_ids=building_ids,
            centroids=centroids,
            scales=scales,
        )

    return collate


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_data_loader(
    dataset: Dataset[PolygonSample],
    *,
    batch_size: int,
    max_vertices: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
) -> DataLoader[Any]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=make_collate_fn(max_vertices),
        worker_init_fn=_seed_worker if num_workers else None,
        generator=generator,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )
