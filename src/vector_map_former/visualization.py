"""Qualitative prediction figures for technical reports and error analysis."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from vector_map_former.data.collate import make_collate_fn
from vector_map_former.data.mapgeneralizer import MapGeneralizerDataset
from vector_map_former.geometry import reconstruct_polygon
from vector_map_former.models.outputs import ModelOutput


def _closed(ring: np.ndarray) -> np.ndarray:
    return np.concatenate([ring, ring[:1]], axis=0)


def create_prediction_figure(
    *,
    model: nn.Module,
    dataset: MapGeneralizerDataset,
    device: torch.device,
    output_path: str | Path,
    examples: int = 9,
) -> Path:
    """Render source, target, and prediction overlays for the first examples."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Visualization requires matplotlib") from error

    if examples < 1:
        raise ValueError("examples must be positive")
    examples = min(examples, len(dataset))
    columns = min(3, examples)
    rows = math.ceil(examples / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4.5 * rows), squeeze=False)
    collate = make_collate_fn(max(dataset[index].coordinates.shape[0] for index in range(examples)))
    model.eval()

    for index in range(examples):
        sample = dataset[index]
        batch = collate([sample]).to(device)
        with torch.inference_mode():
            output = model(
                batch.features,
                coordinates=batch.coordinates,
                padding_mask=batch.padding_mask,
                lengths=batch.lengths,
            )
        if not isinstance(output, ModelOutput):
            raise TypeError("model must return ModelOutput")
        predicted_actions = (
            output.action_logits[0, : len(sample.actions)].argmax(dim=-1).cpu().numpy()
        )
        movement_scale = float(sample.scale.item()) if dataset.normalize_movement else 1.0
        predicted_movement = (
            output.movements[0, : len(sample.actions)].cpu().numpy() * movement_scale
        )
        target_movement = sample.movements.numpy() * movement_scale
        source = sample.raw_coordinates.numpy()
        target = reconstruct_polygon(source, sample.actions.numpy(), target_movement)
        predicted = reconstruct_polygon(source, predicted_actions, predicted_movement)

        axis = axes[index // columns][index % columns]
        source_closed = _closed(source)
        axis.plot(source_closed[:, 0], source_closed[:, 1], "--", color="0.65", label="source")
        axis.scatter(source[:, 0], source[:, 1], s=12, color="0.45")
        if target is not None:
            target_closed = _closed(target)
            axis.plot(target_closed[:, 0], target_closed[:, 1], color="#1b9e77", label="target")
        if predicted is not None:
            predicted_closed = _closed(predicted)
            axis.plot(
                predicted_closed[:, 0],
                predicted_closed[:, 1],
                color="#d95f02",
                label="prediction",
            )
        axis.set_title(f"Building {sample.building_id}")
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(alpha=0.2)
        if index == 0:
            axis.legend(loc="best")

    for index in range(examples, rows * columns):
        axes[index // columns][index % columns].axis("off")
    figure.tight_layout()
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return destination
