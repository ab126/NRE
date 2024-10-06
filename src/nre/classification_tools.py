import copy
import warnings

import numpy as np
from matplotlib import pyplot as plt
from scipy.special import expit, logit
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

marker_rotation = ['o', 'X', 's', '*', 'v', '2']


def perf_measure(y_actual, y_hat, pos_val=1, neg_val=-1):
    """Calculate number of True Positives, False Positives, True Negatives and False Negatives from predictions"""
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for i in range(len(y_hat)):
        if y_actual[i] == y_hat[i] == pos_val:
            tp += 1
        if y_hat[i] == pos_val and y_actual[i] != y_hat[i]:
            fp += 1
        if y_actual[i] == y_hat[i] == neg_val:
            tn += 1
        if y_hat[i] == neg_val and y_actual[i] != y_hat[i]:
            fn += 1

    return tp, fp, tn, fn


def add_disc_points(tpr, fpr, typ='pessimistic'):
    """
    Adjusts fpr or tpr so that ROC curve is discretized. If typ=='pessimistic' points are added below original points
    and if typ == 'optimistic' points are added above original points.
    """
    tpr.append(0)
    fpr.append(0)
    tpr, fpr = np.sort(tpr), np.sort(fpr)
    new_fpr = []
    new_tpr = []
    assert len(tpr) == len(fpr), 'fpr and tpr have different length'
    for i in range(len(tpr)):
        if typ == 'optimistic':
            new_fpr.extend([fpr[i], fpr[i]])
            if i != len(tpr) - 1:
                new_tpr.extend([tpr[i], tpr[i + 1]])
            else:
                new_tpr.extend([tpr[i], 1])
        else:  # typ == 'pessimistic':
            new_tpr.extend([tpr[i], tpr[i]])
            if i != len(tpr) - 1:
                new_fpr.extend([fpr[i], fpr[i + 1]])
            else:
                new_fpr.extend([fpr[i], 1])
    new_tpr.append(1)
    new_fpr.append(1)
    return np.sort(new_tpr), np.sort(new_fpr)


def remove_duplicate_fpr_tpr(fpr, tpr):
    """Given a list of false positive rates remove duplicate values"""
    val_x, val_y = 0, 0
    new_fpr, new_tpr = [0], [0]
    for x, y in zip(fpr, tpr):
        if x != val_x or y != val_y:
            new_fpr.append(x)
            new_tpr.append(y)
            val_x, val_y = x, y
    return np.array(new_fpr), np.array(new_tpr)


def infer_roc(fpr, tpr, n_points=100, deg=1, min_ro_points=5, far_point_mult=1.35, far_point_weight=0.05):
    """
    Given Discrete ROC points infers a smooth ROC curve by fitting a line on logit transform of (fpr, tpr) points
    --------------
    :param fpr: False Positive Rates
    :param tpr: True Positive Rates
    :param n_points: Number of points used for the inferred curve
    :param deg: degree of the polynomial fit used in logit space. (deg=1 is highly recommended)
    :param min_ro_points: Minimum Receiver Operating points except (0, 0) and (1, 1) before adding far points
    :param far_point_mult: Multiplier for points in logit space that correspond to (0, 0) and (1, 1) points in
        fpr, tpr space. The most distant coordinate of points in logit space is multiplied by this parameter to
        determine the placeholder "far points" that correspond to (0, 0) and (1, 1) points in fpr, tpr space.
    :param far_point_weight: Relative Weight of the far points
    :return: Coordinates (x, y_predict) of the inferred ROC curves
    """
    fpr_t, tpr_t = logit(fpr), logit(tpr)
    fpr_t, tpr_t = fpr_t[1:-1], tpr_t[1:-1]  # Non-trivial operating points

    ind_x, ind_y = ~np.isinf(fpr_t), ~np.isinf(tpr_t)
    fpr_op, tpr_op = fpr_t[ind_x], tpr_t[ind_y]  # Non-infinite operating coordinates
    if len(fpr_op) != 0 and len(tpr_op) != 0:
        fpr_mean = np.mean(fpr_op)
        tpr_mean = np.mean(tpr_op)
        far_point_coord = np.max(np.abs(np.concatenate((fpr_op, tpr_op)))) * far_point_mult
    elif len(fpr_op) != 0:  # len(tpr_op) == 0
        fpr_mean = np.mean(fpr_op)
        tpr_mean = 0
        far_point_coord = np.max(fpr_op) * far_point_mult
    elif len(tpr_op) != 0:  # len(fpr_op) == 0
        tpr_mean = np.mean(tpr_op)
        fpr_mean = 0
        far_point_coord = np.max(tpr_op) * far_point_mult
    else:
        fpr_mean = 0
        tpr_mean = 0
        far_point_coord = 1
    far_point_low = (-far_point_coord + fpr_mean, -far_point_coord + tpr_mean)
    far_point_high = (far_point_coord + fpr_mean, far_point_coord + tpr_mean)

    # Replace inf coordinates with the "far points coordinates"
    fpr_t[np.isneginf(fpr_t)] = far_point_low[0]  # -far_point_coord
    fpr_t[np.isposinf(fpr_t)] = far_point_high[0]  # far_point_coord
    tpr_t[np.isneginf(tpr_t)] = far_point_low[1]  # -far_point_coord
    tpr_t[np.isposinf(tpr_t)] = far_point_high[1]  # far_point_coord

    sample_weights = np.ones(len(fpr_t))
    if len(fpr_t) < min_ro_points:  # Add far points
        fpr_t = np.concatenate((np.array([far_point_low[0]]), fpr_t, np.array([far_point_high[0]])))
        tpr_t = np.concatenate((np.array([far_point_low[1]]), tpr_t, np.array([far_point_high[1]])))
        sample_weights = np.concatenate((np.array([far_point_weight]), sample_weights, np.array([far_point_weight])))

    poly = PolynomialFeatures(degree=deg, include_bias=False)  # Bias term is included in LinearRegression instead
    poly_features = poly.fit_transform(fpr_t.reshape(-1, 1))

    poly_reg_model = LinearRegression(fit_intercept=True)
    poly_reg_model.fit(poly_features, tpr_t, sample_weight=sample_weights)

    x_t = logit(np.linspace(0, 1, n_points))
    ind = ~np.isinf(x_t)
    x_t = x_t[ind]
    poly_x = poly.fit_transform(x_t.reshape(-1, 1))
    y_predict_t = poly_reg_model.predict(poly_x)

    x = np.concatenate((np.array([0]), expit(x_t), np.array([1])))
    y_predict = np.concatenate((np.array([0]), expit(y_predict_t), np.array([1])))
    return x, y_predict


def plot_roc_curves(roc_curves, title, smooth_roc=True, show=False, font_size=18):
    """
    Plots the Receiver Operating Characteristic curves of the classification models

    :param roc_curves: Dictionary of model ROC points{model:(fpr, tpr)}
    :param title: Title of the plot
    :param smooth_roc: If True, fits a smooth ROC curve
    :param show: If True, plt.show()
    :param font_size: Fontsize of labels and ticks
    :return fig: figure element
    """

    plt.rcParams.update({'font.size': font_size})
    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    ax.set_title(title)

    for i, mdl in enumerate(roc_curves):
        fpr, tpr = roc_curves[mdl]
        if smooth_roc:
            fpr, tpr = remove_duplicate_fpr_tpr(fpr, tpr)
            x_roc, y_roc = infer_roc(fpr, tpr)
            ax.scatter(fpr, tpr, label=mdl, marker=marker_rotation[i])
            ax.plot(x_roc, y_roc)
        else:
            tpr, fpr = add_disc_points(list(tpr), list(fpr))
            ax.plot(fpr, tpr, label=mdl)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if show:
        plt.show()
    return fig


def max_ba_operating_point(fpr_roc, tpr_roc, eps=0.05, fpr_init=0.1, max_iter=1_000):
    """Calculates numerically the operating point with maximum balanced accuracy from ROC curve points"""
    x_roc = copy.copy(fpr_roc)
    if type(x_roc) != np.ndarray:
        x_roc = np.array(x_roc)
    y_roc = copy.copy(tpr_roc)
    if type(y_roc) != np.ndarray:
        y_roc = np.array(y_roc)

    i = 0
    err = 1
    conv_bool = False
    x_curr = fpr_init
    y_curr = y_roc[_locate_closest(x_roc, x_curr)]
    while err > eps and i < max_iter:
        y_tan = y_curr + (x_roc - x_curr)

        x_inter, y_inter = _find_intersects_numerically(x_roc, y_roc, y_tan, eps=eps / 10)
        if len(x_inter) == 0:
            warnings.warn("No intersection point found.")
            conv_bool = False
            break
        if len(x_inter) == 1:
            x_curr, y_curr = x_inter[0], y_inter[0]
            conv_bool = True
            break
        elif len(x_inter) > 2:
            warnings.warn("Expected 2 intersection points, got {} instead".format(len(x_inter)))
            x_curr = np.mean(x_inter)
            y_curr = y_roc[_locate_closest(x_roc, x_curr)]
            i += 1
            continue
        idx = np.argmax(np.abs(x_inter - x_curr))  # Index of the other point
        x_other, y_other = x_inter[idx], y_inter[idx]

        # Next Guess
        err = np.abs(x_other - x_curr) / 2
        x_curr = (x_curr + x_other) / 2
        y_curr = y_roc[_locate_closest(x_roc, x_curr)]
        i += 1
    if i == max_iter:
        conv_bool = False
    elif err <= eps:
        conv_bool = True
    return x_curr, y_curr, conv_bool


def _locate_closest(x_arr, target):
    """Find the closest point's index to target in x_arr"""
    diff = np.abs(x_arr - target)
    return np.argmin(diff)


def _find_intersects_numerically(xs, y1s, y2s, eps=0.05):
    """
    Given curves points xs (shared), yis; returns the points of intersect via checking closeness of
    respective points
    """
    uniform, sep = _check_uniformity(xs, eps=eps)
    if not uniform:
        raise Exception("x coordinates of curve points must be uniformly separated")

    y_diff = y2s - y1s
    ind = np.abs(y_diff) <= eps
    ind = _non_median_supress_indices(ind)
    x_inter, y_inter = xs[ind], y2s[ind]
    return x_inter, y_inter


def _non_median_supress_indices(indices):
    """Given a list of indices returns continues True entries with only True at the median"""
    sup_indices = np.array([False for _ in indices])
    prev_ind = False
    start_idx = 0
    end_idx = -1
    for i, ind in enumerate(indices):
        if not prev_ind and ind:
            start_idx = i
            prev_ind = True
        elif prev_ind and (not ind or i == len(indices) - 1):
            end_idx = i
            med_idx = (start_idx + end_idx) // 2
            sup_indices[med_idx] = True
            prev_ind = False
    return sup_indices


def _check_uniformity(x_curve, eps=0.001):
    """Given a curve's points' x coordinate, checks if they are uniformly separated and returns the separation"""
    assert len(x_curve) >= 2, "Need more than 1 point to describe a curve"
    x = copy.copy(x_curve)
    if type(x) != np.ndarray:
        x = np.array(x)

    diff = x[1:] - x[:-1]
    values = np.unique(diff)
    if (np.max(values) - np.min(values)) <= eps:
        return True, np.mean(values)
    else:
        return False, None


def get_ba_from_operating_point(fpr_op, tpr_op):
    """Returns the balanced accuracy given the fpr and tpr of the operating point"""
    return 0.5 * (1 - fpr_op + tpr_op)
