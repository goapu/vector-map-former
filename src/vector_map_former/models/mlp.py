"""Independent per-vertex MLP baseline."""

from __future__ import annotations

from torch import Tensor, nn

from vector_map_former.constants import NUM_ACTIONS
from vector_map_former.models.outputs import ModelOutput


class VertexMLP(nn.Module):
    """A local baseline that deliberately has no sequence interaction."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.action_head = nn.Linear(hidden_dim, NUM_ACTIONS)
        self.movement_head = nn.Linear(hidden_dim, 2)

    def forward(
        self,
        features: Tensor,
        *,
        coordinates: Tensor | None = None,
        padding_mask: Tensor | None = None,
        lengths: Tensor | None = None,
    ) -> ModelOutput:
        del coordinates, lengths
        hidden = self.backbone(features)
        if padding_mask is not None:
            hidden = hidden.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return ModelOutput(
            action_logits=self.action_head(hidden),
            movements=self.movement_head(hidden),
        )
