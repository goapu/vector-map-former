import torch

from vector_map_former.data.collate import PolygonBatch
from vector_map_former.losses import MultiTaskCriterion, balanced_class_weights
from vector_map_former.metrics import ClassificationAccumulator
from vector_map_former.models.outputs import ModelOutput


def _batch() -> PolygonBatch:
    return PolygonBatch(
        features=torch.zeros(1, 4, 2),
        coordinates=torch.zeros(1, 4, 2),
        raw_coordinates=torch.zeros(1, 4, 2),
        actions=torch.tensor([[0, 1, 2, -100]]),
        movements=torch.tensor([[[0.0, 0.0], [0.0, 0.0], [0.2, -0.1], [0.0, 0.0]]]),
        padding_mask=torch.tensor([[False, False, False, True]]),
        lengths=torch.tensor([3]),
        building_ids=torch.tensor([1]),
        centroids=torch.zeros(1, 2),
        scales=torch.ones(1),
    )


def test_multitask_loss_masks_padding_and_non_move_vertices() -> None:
    logits = torch.tensor(
        [[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0], [0.0, 0.0, 0.0]]],
        requires_grad=True,
    )
    movements = torch.zeros(1, 4, 2, requires_grad=True)
    criterion = MultiTaskCriterion(class_weights=None, movement_weight=1.0)
    loss = criterion(ModelOutput(logits, movements), _batch())
    assert loss.valid_vertices == 3
    assert loss.moved_vertices == 1
    assert loss.total.item() > 0
    loss.total.backward()


def test_balanced_weights_and_classification_metrics() -> None:
    weights = balanced_class_weights(torch.tensor([10, 20, 40]))
    assert weights[0] > weights[1] > weights[2]
    accumulator = ClassificationAccumulator()
    logits = torch.tensor([[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]])
    targets = torch.tensor([[0, 1, 2]])
    accumulator.update(logits, targets)
    report = accumulator.compute()
    assert report.accuracy == 1.0
    assert report.macro_f1 == 1.0
