import unittest
import numpy as np
from numpy.testing import assert_array_equal

from nre.network_connectivity import ConnectivityUnit


class TestNetworkModel(unittest.TestCase):

    def test_add_entities(self):
        orig_entities = list(range(3))
        cu = ConnectivityUnit(entity_names=orig_entities)
        cu.mat_x = np.array([1, 2, 3])
        cu.mat_p = np.array([[9, 0, -4], [0, 4, 1], [-4, 1, 8]])

        cu.add_entities([6, 7])
        self.assertEqual(cu.entity_names, [0, 1, 2, 6, 7])
        assert_array_equal(cu.mat_f, np.eye(5))
        assert_array_equal(cu.mat_x, np.array([1, 2, 3, 2, 2]))
        assert_array_equal(cu.mat_p, np.array([[9, 0, -4, 0, 0],
                                             [0, 4, 1, 0, 0],
                                             [-4, 1, 8, 0, 0],
                                             [0, 0, 0, 7, 0],
                                             [0, 0, 0, 0, 7]]))


if __name__ == '__main__':
    unittest.main()
