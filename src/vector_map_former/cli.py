"""Command-line interface for auditable data inspection and experiments."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import torch

from vector_map_former.config import load_config, project_config_from_dict
from vector_map_former.data import MapGeneralizerDataset, make_data_loader
from vector_map_former.data.features import feature_dimension
from vector_map_former.data.mapgeneralizer import Split
from vector_map_former.losses import MultiTaskCriterion
from vector_map_former.models import build_model
from vector_map_former.runtime import atomic_json_dump, configure_logging, resolve_device
from vector_map_former.training import evaluate_model, train_project
from vector_map_former.visualization import create_prediction_figure

LOGGER = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vmf",
        description="VectorMapFormer supervised research pipeline",
    )
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="audit MapGeneralizer splits")
    inspect_parser.add_argument("--data-dir", required=True)
    inspect_parser.add_argument("--output")

    train_parser = subparsers.add_parser("train", help="train and test a configured model")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--data-dir")
    train_parser.add_argument(
        "--model",
        choices=["mlp", "circular_cnn", "transformer", "ring_transformer"],
    )
    train_parser.add_argument("--output-dir")
    train_parser.add_argument("--epochs", type=int)
    train_parser.add_argument("--movement-weight", type=float)
    train_parser.add_argument("--device")

    evaluate_parser = subparsers.add_parser("evaluate", help="evaluate a saved checkpoint")
    evaluate_parser.add_argument("--checkpoint", required=True)
    evaluate_parser.add_argument("--data-dir", required=True)
    evaluate_parser.add_argument("--split", choices=["train", "valid", "test"], default="test")
    evaluate_parser.add_argument("--output")
    evaluate_parser.add_argument("--device", default="auto")
    evaluate_parser.add_argument("--no-geometry", action="store_true")

    visualize_parser = subparsers.add_parser("visualize", help="render checkpoint predictions")
    visualize_parser.add_argument("--checkpoint", required=True)
    visualize_parser.add_argument("--data-dir", required=True)
    visualize_parser.add_argument("--split", choices=["train", "valid", "test"], default="test")
    visualize_parser.add_argument("--output", required=True)
    visualize_parser.add_argument("--examples", type=int, default=9)
    visualize_parser.add_argument("--device", default="auto")
    return parser


def _inspect(data_dir: str, output: str | None) -> int:
    datasets = {
        split: MapGeneralizerDataset(
            data_dir,
            cast(Split, split),
            validate_on_load=True,
        )
        for split in ("train", "valid", "test")
    }
    report: dict[str, Any] = {split: dataset.summary() for split, dataset in datasets.items()}
    overlaps: dict[str, int] = {}
    names = tuple(datasets)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            overlaps[f"{first}-{second}"] = len(
                datasets[first].building_ids() & datasets[second].building_ids()
            )
    report["building_id_overlap"] = overlaps
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if output:
        atomic_json_dump(report, output)
    return 0


def _load_checkpoint(
    checkpoint_path: str,
    data_dir: str,
    device_name: str,
) -> tuple[dict[str, Any], Any, torch.device]:
    device = resolve_device(device_name)
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    config_values = dict(checkpoint["config"])
    config_values["data"] = {**config_values["data"], "data_dir": data_dir}
    config = project_config_from_dict(config_values)
    model = build_model(
        config.model,
        input_dim=feature_dimension(config.data.feature_set),
        max_vertices=config.data.max_vertices,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return checkpoint, (config, model), device


def _evaluate(args: argparse.Namespace) -> int:
    checkpoint, (config, model), device = _load_checkpoint(
        args.checkpoint,
        args.data_dir,
        args.device,
    )
    dataset = MapGeneralizerDataset(
        config.data.data_dir,
        args.split,
        feature_set=config.data.feature_set,
        normalize_movement=config.data.normalize_movement,
        validate_on_load=True,
        max_vertices=config.data.max_vertices,
    )
    loader = make_data_loader(
        dataset,
        batch_size=config.training.batch_size,
        max_vertices=config.data.max_vertices,
        shuffle=False,
        seed=config.seed,
        num_workers=config.training.num_workers,
    )
    class_weights = checkpoint.get("class_weights")
    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = MultiTaskCriterion(
        class_weights=class_weights,
        movement_weight=config.training.movement_loss_weight,
    ).to(device)
    report = evaluate_model(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        normalize_movement=config.data.normalize_movement,
        geometry=not args.no_geometry,
        predictions_path=None,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output:
        atomic_json_dump(report, args.output)
    return 0


def _visualize(args: argparse.Namespace) -> int:
    _, (config, model), device = _load_checkpoint(
        args.checkpoint,
        args.data_dir,
        args.device,
    )
    dataset = MapGeneralizerDataset(
        config.data.data_dir,
        args.split,
        feature_set=config.data.feature_set,
        normalize_movement=config.data.normalize_movement,
        validate_on_load=True,
        max_vertices=config.data.max_vertices,
    )
    output = create_prediction_figure(
        model=model,
        dataset=dataset,
        device=device,
        output_path=args.output,
        examples=args.examples,
    )
    LOGGER.info("wrote %s", output)
    return 0


def _train(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.data_dir:
        config = replace(config, data=replace(config.data, data_dir=args.data_dir))
    if args.model:
        config = replace(config, model=replace(config.model, name=args.model))
    if args.output_dir:
        config = replace(config, output=replace(config.output, output_dir=args.output_dir))
    training = config.training
    if args.epochs is not None:
        training = replace(training, epochs=args.epochs)
    if args.movement_weight is not None:
        training = replace(training, movement_loss_weight=args.movement_weight)
    if args.device:
        training = replace(training, device=args.device)
    config = replace(config, training=training)
    config.validate()
    result = train_project(config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    try:
        if args.command == "inspect":
            return _inspect(args.data_dir, args.output)
        if args.command == "train":
            return _train(args)
        if args.command == "evaluate":
            return _evaluate(args)
        if args.command == "visualize":
            return _visualize(args)
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
        LOGGER.error("%s", error)
        return 2
    parser.error(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
