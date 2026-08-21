"""Model factory and public model types."""

from __future__ import annotations

from torch import nn

from vector_map_former.config import ModelConfig
from vector_map_former.models.circular_cnn import CircularCNN
from vector_map_former.models.mlp import VertexMLP
from vector_map_former.models.transformer import VectorTransformer


def build_model(config: ModelConfig, *, input_dim: int, max_vertices: int) -> nn.Module:
    """Construct a model from validated configuration."""

    if config.name == "mlp":
        return VertexMLP(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )
    if config.name == "circular_cnn":
        return CircularCNN(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )
    if config.name in {"transformer", "ring_transformer"}:
        ring_specific = config.name == "ring_transformer"
        return VectorTransformer(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            feedforward_dim=config.feedforward_dim,
            dropout=config.dropout,
            max_vertices=max_vertices,
            relative_bias=ring_specific or config.relative_bias,
            geometric_bias=ring_specific and config.geometric_bias,
            absolute_position=not ring_specific,
        )
    raise ValueError(f"unsupported model: {config.name}")


__all__ = ["CircularCNN", "VectorTransformer", "VertexMLP", "build_model"]
