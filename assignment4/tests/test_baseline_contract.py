import unittest

import numpy as np

from ml441_assignment4.baselines import capped_positions, neighbour_vote


class BaselineContract(unittest.TestCase):
    def test_zero_distance_neighbours_alone_vote(self):
        distances = np.array([[0, 0, 0.001], [1, 1, 3]], dtype=float)
        indices = np.array([[0, 1, 2], [0, 1, 2]])
        target = np.array([2, 8, 8])
        setting = {"k": 3, "weights": "distance", "vote_exponent": 0.5}
        np.testing.assert_array_equal(neighbour_vote(distances, indices, target, [2, 8], setting), [2, 8])

    def test_cap_retains_all_rare_rows_and_is_seeded(self):
        target = np.array([0] * 30 + [1] * 4)
        positions = capped_positions(target, 10, 17, [0, 1])
        self.assertEqual(len(positions), 14)
        self.assertEqual(len(np.unique(positions)), 14)
        self.assertTrue(set(range(30, 34)).issubset(positions))
        np.testing.assert_array_equal(positions, capped_positions(target, 10, 17, [0, 1]))


if __name__ == "__main__":
    unittest.main()
