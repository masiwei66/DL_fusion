import unittest

import torch

from qianhai7_training.data.sampler import RotatingRealizationSampler
from qianhai7_training.contracts import WindowBatch
from qianhai7_training.models.temporal import temporal_features


class RealizationSamplingTests(unittest.TestCase):
    def test_training_sampler_mixes_realizations_within_each_epoch(self):
        window_count = 12
        realization_count = 4
        sampler = RotatingRealizationSampler(
            window_count,
            realization_count,
            seed=123,
        )

        for epoch in range(3):
            sampler.set_epoch(epoch)
            indices = list(iter(sampler))
            self.assertEqual(len(indices), window_count)
            realization_ids = [index // window_count for index in indices]
            self.assertEqual(set(realization_ids), set(range(realization_count)))
            self.assertEqual(
                [realization_ids.count(value) for value in range(realization_count)],
                [3, 3, 3, 3],
            )

    def test_evaluation_sampler_visits_every_window_for_every_realization(self):
        sampler = RotatingRealizationSampler(
            window_count=3,
            realizations=2,
            seed=123,
            shuffle=False,
            strategy="all",
        )
        self.assertEqual(list(sampler), [0, 1, 2, 3, 4, 5])

    @staticmethod
    def _batch(point_counts):
        window_length = 2
        max_points = max(point_counts)
        values = torch.zeros(len(point_counts), window_length, max_points)
        valid_mask = torch.zeros_like(values, dtype=torch.bool)
        point_present_mask = torch.zeros(len(point_counts), max_points, dtype=torch.bool)
        point_features = torch.zeros(len(point_counts), max_points, 2)
        for row, count in enumerate(point_counts):
            values[row, :, :count] = 2.0
            valid_mask[row, :, :count] = True
            point_present_mask[row, :count] = True
        return WindowBatch(
            values=values,
            valid_mask=valid_mask,
            quality=valid_mask.float(),
            point_features=point_features,
            time_features=torch.zeros(len(point_counts), window_length, 2),
            point_present_mask=point_present_mask,
            mapping_distance=torch.zeros(len(point_counts), max_points),
            mapping_valid_mask=point_present_mask,
            temperature_valid_mask=torch.ones(len(point_counts), window_length, 1, dtype=torch.bool),
            teacher_features={"modal_frequencies": torch.zeros(len(point_counts), 1)},
            targets={},
            metadata=[{} for _ in point_counts],
        )

    def test_temporal_valid_ratio_is_independent_of_batch_padding(self):
        single = temporal_features(self._batch([2]))[0]
        padded = temporal_features(self._batch([2, 4]))[0]
        torch.testing.assert_close(single, padded)


if __name__ == "__main__":
    unittest.main()
