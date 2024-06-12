import unittest
import numpy as np
from numpy.testing import assert_array_equal

from src.real_time_model import NetworkModel


class TestNetworkModel(unittest.TestCase):

    def test_add_entities(self):
        orig_entities = list(range(3))
        nm = NetworkModel(entity_names=orig_entities)
        nm.mat_x = np.array([1, 2, 3])
        nm.mat_p = np.array([[9, 0, -4], [0, 4, 1], [-4, 1, 8]])

        nm.add_entities([6, 7])
        self.assertEqual(nm.entity_names, [0, 1, 2, 6, 7])
        assert_array_equal(nm.mat_f, np.eye(5))
        assert_array_equal(nm.mat_x, np.array([1, 2, 3, 2, 2]))
        assert_array_equal(nm.mat_p, np.array([[9, 0, -4, 0, 0],
                                             [0, 4, 1, 0, 0],
                                             [-4, 1, 8, 0, 0],
                                             [0, 0, 0, 7, 0],
                                             [0, 0, 0, 0, 7]]))


if __name__ == '__main__':
    unittest.main()
