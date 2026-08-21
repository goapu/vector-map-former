"""Reproducible training and evaluation orchestration."""

from __future__ import annotations

import logging
import time
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from vector_map_former.config import ProjectConfig
from vector_map_former.data.collate import PolygonBatch, make_data_loader
from vector_map_former.data.features import feature_dimension
from vector_map_former.data.mapgeneralizer import MapGeneralizerDataset
from vector_map_former.geometry import GeometryAccumulator
from vector_map_former.losses import MultiTaskCriterion, balanced_class_weights
from vector_map_former.metrics import ClassificationAccumulator, MovementAccumulator
from vector_map_former.models import build_model
from vector_map_former.models.outputs import ModelOutput
from vector_map_former.runtime import (
    atomic_json_dump,
    atomic_torch_save,
    resolve_device,
    runtime_metadata,
    seed_everything,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EpochLosses:
    total: float
    classification: float
    movement: float


class _LossAccumulator:
    def __init__(self) -> None:
        self.total = 0.0
        self.classification = 0.0
        self.movement = 0.0
        self.valid_vertices = 0
        self.move_vertices = 0

    def update(
        self,
        *,
        total: float,
        classification: float,
        movement: float,
        valid_vertices: int,
        move_vertices: int,
    ) -> None:
        self.total += total * valid_vertices
        self.classification += classification * valid_vertices
        self.movement += movement * move_vertices
        self.valid_vertices += valid_vertices
        self.move_vertices += move_vertices

    def compute(self) -> EpochLosses:
        return EpochLosses(
            total=self.total / max(self.valid_vertices, 1),
            classification=self.classification / max(self.valid_vertices, 1),
            movement=self.movement / max(self.move_vertices, 1),
        )


def _forward(model: nn.Module, batch: PolygonBatch) -> ModelOutput:
    output = model(
        batch.features,
        coordinates=batch.coordinates,
        padding_mask=batch.padding_mask,
        lengths=batch.lengths,
    )
    if not isinstance(output, ModelOutput):
        raise TypeError("model must return ModelOutput")
    return output


def _run_epoch(
    *,
    model: nn.Module,
    loader: DataLoader[Any],
    criterion: MultiTaskCriterion,
    device: torch.device,
    optimizer: AdamW | None,
    gradient_clip_norm: float,
) -> tuple[EpochLosses, dict[str, object]]:
    training = optimizer is not None
    model.train(training)
    losses = _LossAccumulator()
    classification = ClassificationAccumulator()
    movement = MovementAccumulator()

    context = cast(
        AbstractContextManager[None],
        nullcontext() if training else torch.inference_mode(),
    )
    with context:
        for batch in loader:
            if not isinstance(batch, PolygonBatch):
                raise TypeError("loader must emit PolygonBatch")
            batch = batch.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            output = _forward(model, batch)
            loss = criterion(output, batch)
            if optimizer is not None:
                loss.total.backward()
                if gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()

            losses.update(
                total=float(loss.total.detach().item()),
                classification=float(loss.classification.item()),
                movement=float(loss.movement.item()),
                valid_vertices=loss.valid_vertices,
                move_vertices=loss.moved_vertices,
            )
            classification.update(output.action_logits, batch.actions)
            movement.update(output.movements, batch.movements, batch.actions)

    report = classification.compute().to_dict()
    report["movement"] = movement.compute()
    return losses.compute(), report


def _checkpoint_payload(
    *,
    model: nn.Module,
    config: ProjectConfig,
    class_weights: Tensor | None,
    epoch: int,
    validation_macro_f1: float,
) -> dict[str, object]:
    return {
        "format_version": 1,
        "model_state": model.state_dict(),
        "config": config.to_dict(),
        "class_weights": class_weights.cpu() if class_weights is not None else None,
        "epoch": epoch,
        "validation_macro_f1": validation_macro_f1,
        "runtime": runtime_metadata(),
    }


def create_datasets(config: ProjectConfig) -> dict[str, MapGeneralizerDataset]:
    return {
        split: MapGeneralizerDataset(
            config.data.data_dir,
            split,  # type: ignore[arg-type]
            feature_set=config.data.feature_set,
            normalize_movement=config.data.normalize_movement,
            validate_on_load=config.data.validate_on_load,
            max_vertices=config.data.max_vertices,
        )
        for split in ("train", "valid", "test")
    }


def _validate_split_disjointness(datasets: dict[str, MapGeneralizerDataset]) -> None:
    split_names = tuple(datasets)
    for first_index, first in enumerate(split_names):
        for second in split_names[first_index + 1 :]:
            overlap = datasets[first].building_ids() & datasets[second].building_ids()
            if overlap:
                raise ValueError(
                    f"building ID leakage between {first} and {second}: {len(overlap)}"
                )


def train_project(config: ProjectConfig) -> dict[str, object]:
    """Train, select on validation macro-F1, and evaluate the best checkpoint."""

    config.validate()
    seed_everything(config.seed)
    device = resolve_device(config.training.device)
    output_dir = Path(config.output.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(config.to_dict(), output_dir / "resolved_config.json")
    atomic_json_dump(runtime_metadata(), output_dir / "runtime.json")

    datasets = create_datasets(config)
    _validate_split_disjointness(datasets)
    atomic_json_dump(
        {name: dataset.summary() for name, dataset in datasets.items()},
        output_dir / "data_audit.json",
    )
    loaders = {
        split: make_data_loader(
            dataset,
            batch_size=config.training.batch_size,
            max_vertices=config.data.max_vertices,
            shuffle=split == "train",
            seed=config.seed + index,
            num_workers=config.training.num_workers,
        )
        for index, (split, dataset) in enumerate(datasets.items())
    }

    model = build_model(
        config.model,
        input_dim=feature_dimension(config.data.feature_set),
        max_vertices=config.data.max_vertices,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    LOGGER.info("device=%s model=%s parameters=%d", device, config.model.name, parameter_count)

    class_weights = (
        balanced_class_weights(datasets["train"].action_counts()).to(device)
        if config.training.class_weighting == "balanced"
        else None
    )
    criterion = MultiTaskCriterion(
        class_weights=class_weights,
        movement_weight=config.training.movement_loss_weight,
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    history: list[dict[str, object]] = []
    best_macro_f1 = float("-inf")
    stale_epochs = 0
    checkpoint_path = output_dir / "best_model.pt"
    started = time.monotonic()
    for epoch in range(1, config.training.epochs + 1):
        train_losses, train_report = _run_epoch(
            model=model,
            loader=loaders["train"],
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            gradient_clip_norm=config.training.gradient_clip_norm,
        )
        valid_losses, valid_report = _run_epoch(
            model=model,
            loader=loaders["valid"],
            criterion=criterion,
            device=device,
            optimizer=None,
            gradient_clip_norm=config.training.gradient_clip_norm,
        )
        valid_score = valid_report["macro_f1"]
        if not isinstance(valid_score, (int, float)):
            raise TypeError("validation macro-F1 must be numeric")
        valid_macro_f1 = float(valid_score)
        epoch_record: dict[str, object] = {
            "epoch": epoch,
            "train_loss": asdict(train_losses),
            "valid_loss": asdict(valid_losses),
            "train_metrics": train_report,
            "valid_metrics": valid_report,
        }
        history.append(epoch_record)
        atomic_json_dump(history, output_dir / "history.json")
        LOGGER.info(
            "epoch=%d train_loss=%.5f valid_loss=%.5f valid_macro_f1=%.4f",
            epoch,
            train_losses.total,
            valid_losses.total,
            valid_macro_f1,
        )
        if valid_macro_f1 > best_macro_f1 + 1e-6:
            best_macro_f1 = valid_macro_f1
            stale_epochs = 0
            atomic_torch_save(
                _checkpoint_payload(
                    model=model,
                    config=config,
                    class_weights=class_weights,
                    epoch=epoch,
                    validation_macro_f1=valid_macro_f1,
                ),
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= config.training.patience:
                LOGGER.info("early stopping after %d stale epochs", stale_epochs)
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = evaluate_model(
        model=model,
        loader=loaders["test"],
        criterion=criterion,
        device=device,
        normalize_movement=config.data.normalize_movement,
        geometry=True,
        predictions_path=(output_dir / "test_predictions.npz")
        if config.output.save_predictions
        else None,
    )
    checkpoint_epoch = checkpoint["epoch"]
    checkpoint_score = checkpoint["validation_macro_f1"]
    if not isinstance(checkpoint_epoch, int) or not isinstance(checkpoint_score, (int, float)):
        raise TypeError("checkpoint selection metadata has invalid types")
    result = {
        "best_epoch": checkpoint_epoch,
        "best_validation_macro_f1": float(checkpoint_score),
        "test": test_metrics,
        "parameters": parameter_count,
        "elapsed_seconds": time.monotonic() - started,
        "checkpoint": str(checkpoint_path),
    }
    atomic_json_dump(result, output_dir / "result.json")
    return result


def evaluate_model(
    *,
    model: nn.Module,
    loader: DataLoader[Any],
    criterion: MultiTaskCriterion,
    device: torch.device,
    normalize_movement: bool,
    geometry: bool,
    predictions_path: Path | None = None,
) -> dict[str, object]:
    losses, report = _run_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        optimizer=None,
        gradient_clip_norm=1.0,
    )
    result: dict[str, object] = {"loss": asdict(losses), "classification": report}
    geometry_accumulator = GeometryAccumulator() if geometry else None
    prediction_records: dict[str, list[np.ndarray]] = {
        "building_ids": [],
        "lengths": [],
        "actions": [],
        "movements": [],
    }

    model.eval()
    with torch.inference_mode():
        for batch in loader:
            if not isinstance(batch, PolygonBatch):
                raise TypeError("loader must emit PolygonBatch")
            device_batch = batch.to(device)
            output = _forward(model, device_batch)
            predicted_actions = output.action_logits.argmax(dim=-1).cpu().numpy()
            predicted_movements = output.movements.cpu().numpy()
            if predictions_path is not None:
                prediction_records["building_ids"].append(batch.building_ids.numpy())
                prediction_records["lengths"].append(batch.lengths.numpy())
                prediction_records["actions"].append(predicted_actions)
                prediction_records["movements"].append(predicted_movements)
            if geometry_accumulator is None:
                continue
            for index, length_tensor in enumerate(batch.lengths):
                length = int(length_tensor.item())
                movement_scale = float(batch.scales[index].item()) if normalize_movement else 1.0
                geometry_accumulator.update(
                    batch.raw_coordinates[index, :length].numpy(),
                    predicted_actions[index, :length],
                    predicted_movements[index, :length] * movement_scale,
                    batch.actions[index, :length].numpy(),
                    batch.movements[index, :length].numpy() * movement_scale,
                )
    if geometry_accumulator is not None:
        result["geometry"] = geometry_accumulator.compute().to_dict()
    if predictions_path is not None:
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            predictions_path,
            building_ids=np.concatenate(prediction_records["building_ids"], axis=0),
            lengths=np.concatenate(prediction_records["lengths"], axis=0),
            actions=np.concatenate(prediction_records["actions"], axis=0),
            movements=np.concatenate(prediction_records["movements"], axis=0),
        )
    return result
