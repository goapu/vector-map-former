import numpy as np
import pytest
import torch

from vector_map_former.constants import PAD_ACTION
from vector_map_former.data import MapGeneralizerDataset, make_collate_fn


def test_dataset_contract(tiny_data_dir) -> None:
    dataset = MapGeneralizerDataset(
        tiny_data_dir,
        "train",
        feature_set="xy_structural",
        max_vertices=8,
    )
    sample = dataset[0]
    assert sample.features.shape == (4, 11)
    assert sample.actions.tolist() == [0, 1, 2, 0]
    assert torch.isfinite(sample.features).all()
    observed_actions = dataset.action_counts().sum().item()
    expected_actions = sum(len(dataset[i].actions) for i in range(len(dataset)))
    assert observed_actions == expected_actions


def test_collate_masks_padding(tiny_data_dir) -> None:
    dataset = MapGeneralizerDataset(tiny_data_dir, "train", max_vertices=8)
    batch = make_collate_fn(8)([dataset[0], dataset[1]])
    assert batch.features.shape == (2, 8, 2)
    assert batch.lengths.tolist() == [4, 5]
    assert batch.padding_mask[0, :4].tolist() == [False] * 4
    assert batch.padding_mask[0, 4:].tolist() == [True] * 4
    assert torch.all(batch.actions[0, 4:] == PAD_ACTION)


def test_fractional_action_labels_are_rejected(tiny_data_dir) -> None:
    path = tiny_data_dir / "vertex_train.npy"
    polygons = np.load(path, allow_pickle=True)
    polygons[0][0, 10] = 1.5
    np.save(path, polygons)
    with pytest.raises(ValueError, match="non-integer action"):
        MapGeneralizerDataset(tiny_data_dir, "train")
