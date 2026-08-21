import numpy as np

from vector_map_former.geometry import GeometryAccumulator, reconstruct_polygon


def test_reconstruction_removes_and_moves_in_order() -> None:
    coordinates = np.asarray([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
    actions = np.asarray([1, 2, 0, 1])
    movements = np.zeros((4, 2))
    movements[1] = [0.1, -0.1]
    result = reconstruct_polygon(coordinates, actions, movements)
    assert result is not None
    assert result.shape == (3, 2)
    np.testing.assert_allclose(result[0], coordinates[0])
    np.testing.assert_allclose(result[-1], coordinates[-1])


def test_reconstruction_rejects_underdefined_polygon() -> None:
    coordinates = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    result = reconstruct_polygon(coordinates, np.asarray([1, 0, 0]), np.zeros((3, 2)))
    assert result is None


def test_geometry_report_separates_invalid_targets_from_predictions() -> None:
    coordinates = np.asarray([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
    movements = np.zeros((4, 2))
    accumulator = GeometryAccumulator()
    accumulator.update(
        coordinates,
        np.asarray([1, 1, 1, 1]),
        movements,
        np.asarray([0, 0, 1, 1]),
        movements,
    )
    report = accumulator.compute()
    assert report.invalid_targets == 1
    assert report.invalid_predictions == 0
    assert report.valid_percentage == 0.0
