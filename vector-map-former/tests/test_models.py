import pytest
import torch

from vector_map_former.models.circular_cnn import CircularCNN
from vector_map_former.models.mlp import VertexMLP
from vector_map_former.models.transformer import VectorTransformer


@pytest.mark.parametrize("model_name", ["mlp", "cnn", "transformer", "ring"])
def test_model_output_shapes(model_name: str) -> None:
    if model_name == "mlp":
        model = VertexMLP(input_dim=2, hidden_dim=16, num_layers=2, dropout=0.0)
    elif model_name == "cnn":
        model = CircularCNN(input_dim=2, hidden_dim=16, num_layers=2, dropout=0.0)
    else:
        ring = model_name == "ring"
        model = VectorTransformer(
            input_dim=2,
            hidden_dim=16,
            num_layers=2,
            num_heads=4,
            feedforward_dim=32,
            dropout=0.0,
            max_vertices=8,
            relative_bias=ring,
            geometric_bias=ring,
            absolute_position=not ring,
        )
    features = torch.randn(2, 8, 2)
    coordinates = features.clone()
    lengths = torch.tensor([5, 8])
    padding = torch.arange(8)[None, :] >= lengths[:, None]
    output = model(
        features,
        coordinates=coordinates,
        padding_mask=padding,
        lengths=lengths,
    )
    assert output.action_logits.shape == (2, 8, 3)
    assert output.movements.shape == (2, 8, 2)
    assert torch.isfinite(output.action_logits).all()


def test_ring_transformer_is_cyclic_shift_equivariant() -> None:
    torch.manual_seed(4)
    model = VectorTransformer(
        input_dim=2,
        hidden_dim=16,
        num_layers=2,
        num_heads=4,
        feedforward_dim=32,
        dropout=0.0,
        max_vertices=8,
        relative_bias=True,
        geometric_bias=True,
        absolute_position=False,
    ).eval()
    features = torch.randn(1, 6, 2)
    padding = torch.zeros(1, 6, dtype=torch.bool)
    lengths = torch.tensor([6])
    original = model(
        features,
        coordinates=features,
        padding_mask=padding,
        lengths=lengths,
    ).action_logits
    shifted_features = torch.roll(features, shifts=2, dims=1)
    shifted = model(
        shifted_features,
        coordinates=shifted_features,
        padding_mask=padding,
        lengths=lengths,
    ).action_logits
    torch.testing.assert_close(
        torch.roll(original, shifts=2, dims=1),
        shifted,
        atol=1e-5,
        rtol=1e-5,
    )


def test_ring_transformer_ignores_extra_padding() -> None:
    torch.manual_seed(8)
    model = VectorTransformer(
        input_dim=2,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        feedforward_dim=32,
        dropout=0.0,
        max_vertices=8,
        relative_bias=True,
        geometric_bias=True,
        absolute_position=False,
    ).eval()
    valid = torch.randn(1, 5, 2)
    short = model(
        valid,
        coordinates=valid,
        padding_mask=torch.zeros(1, 5, dtype=torch.bool),
        lengths=torch.tensor([5]),
    ).action_logits
    padded = torch.cat([valid, torch.randn(1, 3, 2)], dim=1)
    mask = torch.tensor([[False] * 5 + [True] * 3])
    long = model(
        padded,
        coordinates=padded,
        padding_mask=mask,
        lengths=torch.tensor([5]),
    ).action_logits
    torch.testing.assert_close(short, long[:, :5], atol=1e-5, rtol=1e-5)
