"""Circular one-dimensional CNN baseline for closed polygon rings."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from vector_map_former.constants import NUM_ACTIONS
from vector_map_former.models.outputs import ModelOutput


class _CircularResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.convolution = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=0)
        self.normalization = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: Tensor, lengths: Tensor, padding_mask: Tensor) -> Tensor:
        output = torch.zeros_like(hidden)
        for batch_index in range(hidden.shape[0]):
            length = int(lengths[batch_index].item())
            sequence = hidden[batch_index, :length].transpose(0, 1).unsqueeze(0)
            convolved = self.convolution(functional.pad(sequence, (1, 1), mode="circular"))
            output[batch_index, :length] = convolved.squeeze(0).transpose(0, 1)
        hidden = hidden + self.dropout(functional.gelu(output))
        hidden = self.normalization(hidden)
        return hidden.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class CircularCNN(nn.Module):
    """A length-safe circular CNN; padded tokens never enter a ring convolution."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [_CircularResidualBlock(hidden_dim, dropout) for _ in range(num_layers)]
        )
        self.action_head = nn.Linear(hidden_dim, NUM_ACTIONS)
        self.movement_head = nn.Linear(hidden_dim, 2)

    def forward(
        self,
        features: Tensor,
        *,
        coordinates: Tensor | None = None,
        padding_mask: Tensor,
        lengths: Tensor,
    ) -> ModelOutput:
        del coordinates
        hidden = self.input_projection(features)
        hidden = hidden.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        for block in self.blocks:
            hidden = block(hidden, lengths, padding_mask)
        return ModelOutput(
            action_logits=self.action_head(hidden),
            movements=self.movement_head(hidden),
        )
