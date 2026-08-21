"""Vanilla and cyclic-relative Transformer encoders for polygon rings."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from vector_map_former.constants import NUM_ACTIONS
from vector_map_former.models.outputs import ModelOutput


class PairwiseAttentionBias(nn.Module):
    """Per-head ring-distance and optional geometric-distance attention bias."""

    def __init__(
        self,
        *,
        num_heads: int,
        max_vertices: int,
        geometric_bias: bool,
        geometric_bins: int = 16,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.max_vertices = max_vertices
        self.geometric_bias = geometric_bias
        self.geometric_bins = geometric_bins
        self.ring_embedding = nn.Embedding(max_vertices + 1, num_heads)
        self.geometry_embedding = (
            nn.Embedding(geometric_bins, num_heads) if geometric_bias else None
        )
        nn.init.zeros_(self.ring_embedding.weight)
        if self.geometry_embedding is not None:
            nn.init.zeros_(self.geometry_embedding.weight)

    def forward(
        self,
        coordinates: Tensor,
        lengths: Tensor,
        padding_mask: Tensor,
    ) -> Tensor:
        batch_size, sequence_length, _ = coordinates.shape
        positions = torch.arange(sequence_length, device=coordinates.device)
        direct = (positions[:, None] - positions[None, :]).abs()
        direct = direct.unsqueeze(0).expand(batch_size, -1, -1)
        ring_lengths = lengths[:, None, None]
        cyclic = torch.minimum(direct, (ring_lengths - direct).abs())
        cyclic = cyclic.clamp(min=0, max=self.max_vertices).long()
        bias = self.ring_embedding(cyclic).permute(0, 3, 1, 2)

        if self.geometry_embedding is not None:
            distances = torch.cdist(coordinates, coordinates)
            buckets = torch.clamp(
                (distances * self.geometric_bins / 2.0).long(),
                min=0,
                max=self.geometric_bins - 1,
            )
            geometry = self.geometry_embedding(buckets).permute(0, 3, 1, 2)
            bias = bias + geometry

        key_padding = padding_mask[:, None, None, :]
        bias = bias.masked_fill(key_padding, torch.finfo(bias.dtype).min)
        reshaped: Tensor = bias.reshape(
            batch_size * self.num_heads,
            sequence_length,
            sequence_length,
        )
        return reshaped


class RingTransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.feedforward_norm = nn.LayerNorm(hidden_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        hidden: Tensor,
        *,
        padding_mask: Tensor,
        attention_bias: Tensor | None,
    ) -> Tensor:
        normalized = self.attention_norm(hidden)
        if attention_bias is None:
            attended, _ = self.attention(
                normalized,
                normalized,
                normalized,
                key_padding_mask=padding_mask,
                need_weights=False,
            )
        else:
            attended, _ = self.attention(
                normalized,
                normalized,
                normalized,
                attn_mask=attention_bias,
                need_weights=False,
            )
        hidden = hidden + self.attention_dropout(attended)
        hidden = hidden + self.feedforward(self.feedforward_norm(hidden))
        return hidden.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class VectorTransformer(nn.Module):
    """Transformer with optional cyclic-relative and geometric attention bias."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float,
        max_vertices: int,
        relative_bias: bool,
        geometric_bias: bool,
        absolute_position: bool,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.position_embedding = (
            nn.Embedding(max_vertices, hidden_dim) if absolute_position else None
        )
        self.pairwise_bias = (
            PairwiseAttentionBias(
                num_heads=num_heads,
                max_vertices=max_vertices,
                geometric_bias=geometric_bias,
            )
            if relative_bias
            else None
        )
        self.blocks = nn.ModuleList(
            [
                RingTransformerBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    feedforward_dim=feedforward_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.action_head = nn.Linear(hidden_dim, NUM_ACTIONS)
        self.movement_head = nn.Linear(hidden_dim, 2)

    def forward(
        self,
        features: Tensor,
        *,
        coordinates: Tensor,
        padding_mask: Tensor,
        lengths: Tensor,
    ) -> ModelOutput:
        hidden = self.input_projection(features)
        if self.position_embedding is not None:
            positions = torch.arange(features.shape[1], device=features.device)
            hidden = hidden + self.position_embedding(positions)[None, :, :]
        hidden = hidden.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        attention_bias = (
            self.pairwise_bias(coordinates, lengths, padding_mask)
            if self.pairwise_bias is not None
            else None
        )
        for block in self.blocks:
            hidden = block(
                hidden,
                padding_mask=padding_mask,
                attention_bias=attention_bias,
            )
        hidden = self.final_norm(hidden)
        hidden = hidden.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return ModelOutput(
            action_logits=self.action_head(hidden),
            movements=self.movement_head(hidden),
        )
