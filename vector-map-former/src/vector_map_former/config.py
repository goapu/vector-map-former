"""Strict configuration loading for training and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, TypeVar, cast

import yaml

from vector_map_former.constants import FEATURE_SETS, MODEL_NAMES

T = TypeVar("T")


def _strict_dataclass(cls: type[T], values: Any, section: str) -> T:
    if not isinstance(values, dict):
        raise ValueError(f"Configuration section '{section}' must be a mapping")
    allowed = {field.name for field in fields(cast(Any, cls))}
    unknown = set(values) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown keys in '{section}': {names}")
    return cls(**values)


@dataclass(frozen=True, slots=True)
class DataConfig:
    data_dir: str = "data/raw/mapgeneralizer/data/input"
    feature_set: str = "xy"
    max_vertices: int = 32
    normalize_movement: bool = True
    validate_on_load: bool = True

    def validate(self) -> None:
        if self.feature_set not in FEATURE_SETS:
            raise ValueError(f"feature_set must be one of {sorted(FEATURE_SETS)}")
        if self.max_vertices < 3:
            raise ValueError("max_vertices must be at least 3")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    name: str = "mlp"
    hidden_dim: int = 96
    num_layers: int = 3
    dropout: float = 0.1
    num_heads: int = 4
    feedforward_dim: int = 256
    relative_bias: bool = False
    geometric_bias: bool = False

    def validate(self) -> None:
        if self.name not in MODEL_NAMES:
            raise ValueError(f"model.name must be one of {sorted(MODEL_NAMES)}")
        if self.hidden_dim <= 0 or self.num_layers <= 0:
            raise ValueError("hidden_dim and num_layers must be positive")
        if self.feedforward_dim <= 0 or self.num_heads <= 0:
            raise ValueError("feedforward_dim and num_heads must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.name in {"transformer", "ring_transformer"} and (
            self.hidden_dim % self.num_heads != 0
        ):
            raise ValueError("hidden_dim must be divisible by num_heads")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    batch_size: int = 128
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    movement_loss_weight: float = 0.0
    class_weighting: str = "balanced"
    gradient_clip_norm: float = 1.0
    patience: int = 10
    num_workers: int = 0
    device: str = "auto"

    def validate(self) -> None:
        if self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError("batch_size and epochs must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer settings")
        if self.movement_loss_weight < 0:
            raise ValueError("movement_loss_weight must be non-negative")
        if self.class_weighting not in {"balanced", "none"}:
            raise ValueError("class_weighting must be 'balanced' or 'none'")
        if self.patience < 1 or self.num_workers < 0:
            raise ValueError("patience must be positive and num_workers non-negative")
        if self.gradient_clip_norm < 0:
            raise ValueError("gradient_clip_norm must be non-negative")


@dataclass(frozen=True, slots=True)
class OutputConfig:
    output_dir: str = "outputs/baseline"
    save_predictions: bool = True


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    seed: int
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    output: OutputConfig

    def validate(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        self.data.validate()
        self.model.validate()
        self.training.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> ProjectConfig:
    """Load a YAML configuration and reject unknown or invalid keys."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a mapping")

    expected = {"seed", "data", "model", "training", "output"}
    unknown = set(raw) - expected
    missing = expected - set(raw)
    if unknown:
        raise ValueError(f"Unknown top-level keys: {', '.join(sorted(unknown))}")
    if missing:
        raise ValueError(f"Missing top-level keys: {', '.join(sorted(missing))}")

    return project_config_from_dict(raw)


def project_config_from_dict(raw: dict[str, Any]) -> ProjectConfig:
    """Construct and validate configuration from a plain mapping."""

    expected = {"seed", "data", "model", "training", "output"}
    unknown = set(raw) - expected
    missing = expected - set(raw)
    if unknown:
        raise ValueError(f"Unknown top-level keys: {', '.join(sorted(unknown))}")
    if missing:
        raise ValueError(f"Missing top-level keys: {', '.join(sorted(missing))}")
    config = ProjectConfig(
        seed=int(raw["seed"]),
        data=_strict_dataclass(DataConfig, raw["data"], "data"),
        model=_strict_dataclass(ModelConfig, raw["model"], "model"),
        training=_strict_dataclass(TrainingConfig, raw["training"], "training"),
        output=_strict_dataclass(OutputConfig, raw["output"], "output"),
    )
    config.validate()
    return config
