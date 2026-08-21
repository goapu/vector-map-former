import numpy as np

from vector_map_former.data.features import normalize_polygon, structural_features


def test_normalization_is_translation_and_scale_invariant() -> None:
    coordinates = np.asarray([[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [0.0, 2.0]])
    normalized, _, _ = normalize_polygon(coordinates)
    transformed, _, _ = normalize_polygon(coordinates * 7.0 + np.asarray([100.0, -30.0]))
    np.testing.assert_allclose(normalized, transformed, atol=1e-6)


def test_structural_features_are_cyclic() -> None:
    coordinates = np.asarray(
        [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]],
        dtype=np.float32,
    )
    features = structural_features(coordinates)
    rolled = structural_features(np.roll(coordinates, 1, axis=0))
    np.testing.assert_allclose(np.roll(features, 1, axis=0), rolled, atol=1e-6)
    assert features.shape == (4, 11)
