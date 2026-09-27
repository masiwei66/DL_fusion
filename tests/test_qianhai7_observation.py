import tempfile
import unittest
from pathlib import Path

import numpy as np

from qianhai7_training.data.normalization import NormalizationStats
from qianhai7_training.data.observation import build_observation_map


class ObservationTests(unittest.TestCase):
    def test_quad_center_uses_equal_weights(self):
        coords = np.array([[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]], dtype=np.float32)
        mapping = build_observation_map(
            np.array([[1, 1, 0]], dtype=np.float32),
            coords,
            np.array([10, 11, 12, 13]),
            np.array([[10, 11, 12, 13]]),
            tolerance=1e-6,
        )
        np.testing.assert_allclose(mapping.weights[0], 0.25)
        np.testing.assert_allclose(mapping.interpolate(np.array([0, 1, 2, 3], dtype=np.float32)), 1.5)
        self.assertTrue(mapping.mapping_valid[0])

    def test_outside_point_is_retained_but_invalid(self):
        coords = np.array([[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]], dtype=np.float32)
        mapping = build_observation_map(
            np.array([[3, 1, 0]], dtype=np.float32),
            coords,
            np.array([10, 11, 12, 13]),
            np.array([[10, 11, 12, 13]]),
        )
        self.assertFalse(mapping.mapping_valid[0])
        self.assertTrue(np.isfinite(mapping.mapping_distance[0]))


class NormalizationTests(unittest.TestCase):
    def test_stats_round_trip(self):
        stats = NormalizationStats(1, 2, 3, 4, 5, 6, (7, 8), (9, 10), ("state_a",))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stats.json"
            stats.save(path)
            loaded = NormalizationStats.load(path)
        self.assertEqual(loaded, stats)
        np.testing.assert_allclose(loaded.transform_values(np.array([1, 3])), [0, 1])
