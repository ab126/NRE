import unittest
import numpy as np
from matplotlib import pyplot as plt
from numpy.testing import assert_array_equal, assert_array_almost_equal

from src.analyze_cic_ids import infer_roc
from src.classification_tools import _check_uniformity, _non_median_supress_indices, _find_intersects_numerically, \
    max_ba_operating_point, plot_roc_curves


class TestMaxBAOperatingPoint(unittest.TestCase):

    def test_check_uniformity(self):
        x_curve = np.linspace(1, 112, 1110)
        check, _ = _check_uniformity(x_curve)
        self.assertEqual(check, True)
        x_curve[1] = 1.15
        check, _ = _check_uniformity(x_curve)
        self.assertEqual(check, False)

    def test_non_median_supress_indices(self):
        indices = [False, False, False, True, False, False, False]
        gt_indices = [False, False, False, True, False, False, False]
        ret_indices = _non_median_supress_indices(indices)
        assert_array_equal(ret_indices, gt_indices)

        indices = [False, False, False, True, True, True, False, False, False]
        gt_indices = [False, False, False, False, True, False, False, False, False]
        ret_indices = _non_median_supress_indices(indices)
        assert_array_equal(ret_indices, gt_indices)

        indices = [True, True, True, True, True, True, True, True, True]
        gt_indices = [False, False, False, False, True, False, False, False, False]
        ret_indices = _non_median_supress_indices(indices)
        assert_array_equal(ret_indices, gt_indices)

    def test_find_intersects_numerically(self):
        x1 = np.linspace(1, 10, 1000)
        y1 = np.sin(x1)
        y2 = np.cos(x1 / 2)
        x_inter, y_inter = _find_intersects_numerically(x1, y1, y2)
        assert_array_almost_equal(x_inter, np.array([1.04504505, 3.14414414, 5.23423423, 9.42342342]), decimal=2)
        assert_array_almost_equal(y_inter, np.array([8.66563029e-01, -1.27574493e-03, -8.65586691e-01, -6.77268621e-04])
                                  , decimal=2)
        plt.plot(x1, y1, label='1')
        plt.plot(x1, y2, label='2')
        for x, y in zip(x_inter, y_inter):
            plt.scatter(x, y)

    def test_max_ba_operating_point(self):  # TODO: Gotta make it more robust before the sweep
        fpr_roc = np.linspace(0, 1, 1000)
        tpr_roc = np.power(fpr_roc, 1)  # 2
        plt.plot(fpr_roc, tpr_roc)

        x_curr, y_curr = max_ba_operating_point(fpr_roc, tpr_roc)
        y_tan = y_curr + (fpr_roc - x_curr)
        plt.plot(fpr_roc, y_tan)
        plt.show()
        # assert_array_almost_equal([x_curr, y_curr], [0.25625625625625625, 0.5052666545333336], decimal=2)


class TestROCCurves(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        plt.rcParams.update({'font.size': 14})

    @classmethod
    def tearDownClass(cls) -> None:
        pass

    def setUp(self):
        self.curves = {'Near Perfect': ([0., 0., 0., 0.79487179, 0.84615385, 1.], [0., 0.06666667, 1., 1., 1., 1.]),
                       'Side': ([0, 0, 1], [0, 0.8, 1])}
        self.fpr_list = [[0, 0.3, 0.5, 0.6, 1], [0, 0.3, 0.6, 1], [0, 0.3, 1], [0, 1], [0, 0, 1]]
        self.tpr_list = [[0, 0.6, 0.7, 0.8, 1], [0, 0.75, 0.7, 1], [0, 0.9, 1], [0, 1], [0, 0.8, 1]]

    def tearDown(self):
        pass

    def test_infer_roc(self):
        """Test for infer_roc"""
        for fpr, tpr in zip(self.fpr_list, self.tpr_list):
            fig, ax = plt.subplots()  # Create a figure containing a single axes.
            ax.set_title('Test ROC Curve')
            x_roc, y_roc = infer_roc(fpr, tpr)
            ax.scatter(fpr, tpr)
            ax.plot(x_roc, y_roc)
            plt.show()

    def test_plot_roc_curves(self):
        plot_roc_curves(self.curves, 'Test ROC Curves')
        plt.show()


if __name__ == '__main__':
    unittest.main()
