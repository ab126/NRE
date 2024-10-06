import warnings

import numpy as np
import numpy.random as nr
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import scipy
from heapq import heappop, heappush
from itertools import count

from scipy.stats import multivariate_normal as mvn

from .network_connectivity import get_all_entities


def communication_graph_from_df(df, entity_names=None, keep_outsiders=True, src_id_col=' Source IP',
                                dst_id_col=' Destination IP', spanning_tree=False):
    """
    Given canonical DataFrame df, returns the observed communication graph that holds the entities and available
    communication channels between them.

    :param df: Canonical DataFrame where each row is a flow and columns include timestamp, entity identifiers, and
        flow features.
    :param entity_names: The List of entity names that the reading and sample computation is limited to. If None,
        computes for every entity.
    :param keep_outsiders: If True, keeps the entities that have communicated with the ones in entity_names but does not
        belong to entity_names
    :param src_id_col: Column name of the flows data DatatFrame that identifies the source entity.
    :param dst_id_col: Column name of the flows data DatatFrame that identifies the destination entity.
    :param spanning_tree: If True returns only the spanning tree
    :return: NetworkX graph g: Communication graph of the network
    """
    if entity_names:
        if keep_outsiders:
            idx = df[src_id_col].isin(entity_names) | df[dst_id_col].isin(entity_names)
        else:
            idx = df[src_id_col].isin(entity_names) & df[dst_id_col].isin(entity_names)
        df = df[idx].copy()

    g = nx.Graph()
    g.add_nodes_from(entity_names)
    for row in df.to_dict('records'):
        src = row[src_id_col]
        dst = row[dst_id_col]
        g.add_edge(src, dst)
    if spanning_tree:
        g = nx.minimum_spanning_tree(g)
    return g


def safest_path(g, source, risk, target=None, pred=None, path=None):
    """
    Dijkstra's algorithm with generalized distances to find safest paths.
    -------------
    :param g: NetworkX graph
    :param source: Source entity
        All safest paths from this entity to every other entity is calculated
    :param risk: List or array of risks that is used in conjunction with distance type to judge safety
        Risks must be ordered the same way as G.nodes
    :param target: Target entity
        Search is halted if target entity is reached
    :param pred: Dictionary of predecessors
        Updated in place. pred[u] is the entity before u in the safest path.
    :param path: Dictionary of paths
        Updated in place. paths[u] is the safest paths from source to u.
    :return distances: Dictionary of final distances to the source
    """

    distances = {}
    dist_heap = []
    c = count()
    src_risk = risk[source]
    heappush(dist_heap, (src_risk, c, source))
    # dist_heap is a heap with distances as keys.
    # Use dec to mitigate entity comparison
    seen = {source: src_risk}

    while dist_heap:
        d, _, v = heappop(dist_heap)

        if v in distances:  # Reached optimally already
            continue
        distances[v] = d
        if v == target:
            break

        for u in g[v]:
            d_u_new = max(d, risk[u])  # TODO: Add general distance function here

            if u in distances:  # in final distances
                d_u_prev = distances[u]
                if d_u_new < d_u_prev:
                    raise ValueError('Contradictory path:')
                elif pred is not None and d_u_new == d_u_prev:
                    pred[u].append(v)
            elif u not in seen or d_u_new < seen[u]:  # Update neighbor's distances
                seen[u] = d_u_new
                heappush(dist_heap, (d_u_new, next(c), u))
                if pred is not None:
                    pred[u] = [v]
                if path is not None:
                    path[u] = path[v] + [u]
            elif d_u_new == seen[u]:  # Equal risk paths
                if pred is not None:
                    pred[u].append(v)
                if path is not None:
                    warnings.warn('Multi-safest-paths are not added to variable path!')
    return distances


def naive_mc_max(mean, cov, n_sample=1000, n_trial=100):
    """
    Calculates the expected value of multivariate gaussian Random variables given by their
    mean and covariance matrix using simple monte carlo sampling


    :param mean: Mean vector (n,) 1D array
    :param cov: Covariance matrix (n, n) 2D array
    :param n_sample: Number of independent samples at each trial
    :param n_trial: Number of times the calculation is repeated
    :return: res_df: pandas DataFrame containing max of each sample at each trial
    """
    res_df = pd.DataFrame()

    for i in range(n_trial):
        zs = nr.multivariate_normal(mean, cov, n_sample)
        max_z = np.max(zs, 1)
        temp_df = pd.DataFrame({'trial': i, 'sample': [j for j in range(n_sample)], 'max': list(max_z)})
        res_df = pd.concat((res_df, temp_df), ignore_index=True)
    return res_df


def remove_elem_i(mat, i):
    """
    Removes i_th element from the 1D array or i_th flow_row and column from 2D array
    -------------
    :param mat: 1 or 2 dimensional array whose to be reduced
    :param i: Index that is removed
    :return mat_reduced: Reduced array
    """
    assert len(mat.shape) == 1 or len(mat.shape) == 2, "Only implemented for 1D or 2D arrays"
    ind = np.arange(mat.shape[0]) != i
    if len(mat.shape) == 2:
        ind = ind.reshape((-1, 1))
        ind = np.dot(ind, ind.T)
    mat_reduced = mat[ind]
    if len(mat.shape) == 2:
        mat_reduced = mat_reduced.reshape((mat.shape[0] - 1, mat.shape[0] - 1))
    return mat_reduced


def is_pos_def(x):
    """
    Returns True if x is positive definite
    ----------
    :param x: 2D array, matrix
    :return: Boolean indicator of x being positive definite
    """
    return np.all(np.linalg.eigvals(x) > 0)


def get_a_i(cov, mu):
    """
    Given the covariance matrix cov and mean vector mu, returns matrix a in Afonja (1972) Theorem 1
    -------------
    :param cov: Covariance Matrix of original multivariate gaussian random variables x_i's
    :param mu: Mean of original multivariate gaussian random variables x_i's
    :return mat_a: Matrix a whose ith flow_row is the boundary for integrating truncated PDF in Afonja (1972)
    """
    m = cov.shape[0]
    assert len(mu) == m, "Input sizes dont match"
    mu_res = mu.reshape((-1, 1))
    mean_diff = mu_res.T - mu_res
    sigma_ii = np.dot(np.diag(cov).reshape((m, 1)), np.ones((1, m)))
    sigma_jj = np.dot(np.ones((m, 1)), np.diag(cov).reshape((1, m)))
    var_xi_xj = sigma_ii - 2 * cov + sigma_jj

    ind = np.ones((m, m)) == 1
    np.fill_diagonal(ind, False)
    mat_a = np.divide(mean_diff, np.sqrt(var_xi_xj), where=ind)
    np.fill_diagonal(mat_a, -np.inf)
    return mat_a


def cov2corr_coef(cov):
    """
    Given a covariance matrix returns the respective correlation coefficient matrix R, aka. normalized covariance matrix
    --------------
    :param cov: Valid covariance matrix
    :return mat_r: Pearson Correlation Coefficient Matrix
    """
    m = cov.shape[0]
    stdevs = np.sqrt(np.diag(cov).reshape((-1, 1)))
    mat_r = cov / ((stdevs @ np.ones((1, m))) * (np.ones((m, 1)) @ stdevs.T))
    return mat_r


def get_r_i_hat(cov, mu, i):
    """
    Given the covariance matrix cov, mean vector mu and index i returns the correlation matrix R_i in Afonja (1972)
    Theorem 1.
    -------------
    :param cov: Covariance Matrix of original multivariate gaussian random variables x_i's
    :param mu: Mean of original multivariate gaussian random variables x_i's
    :param i: Index i for calculating R_i in Afonja (1972) which corresponds to instances where i_th component of x is
        the max
    :return mat_r_i_hat: Correlation matrix R_i whose j,j' element is corr(x_i - x_j, x_i - x_j')
    """
    m = cov.shape[0]
    assert len(mu) == m, "Input sizes dont match"
    # mu_res = mu.reshape((m, 1))
    # mean_term = (mu[i] * np.ones((m, 1)) - mu_res) @ (mu[i] * np.ones((m, 1)) - mu_res).T
    cov_term = cov[i, i] * np.ones((m, m)) - np.ones((m, 1)) @ cov[i, :].reshape((1, -1)) - cov[:, i].reshape(
        (-1, 1)) @ np.ones((1, m)) + cov
    mat_r_i_hat = cov2corr_coef(
        remove_elem_i(cov_term, i))  # + mean_term $ r is interpretted as Pearson Correlation Coefficient
    return mat_r_i_hat


def count2bin_digits(dec, num_digits=None):
    """
    Given a decimal number returns its binary digits
    ---------
    :param dec: A decimal number
    :param num_digits: Number of binary digits of output. Output is truncated or zeros padded if necessary
    :return: String corresponding to binary representation of input dec
    """
    if num_digits is None:
        num_digits = int(np.ceil(np.log(dec) / np.log(2)))

    bin_str = bin(dec).split('0b')[1]
    if len(bin_str) > num_digits:
        return bin_str[:num_digits]
    return ''.join(['0' for _ in range(num_digits - len(bin_str))]) + bin_str


def reverse_mvn_cdf(mat_r, b):
    """
    Given the covariance matrix mat_r and vector b returns the probability
    of multivariate gaussian random variable being larger than b for each component, aka P(X_i > b_i, for all i)
    -------------
    :param mat_r: Covariance Matrix for the multivariate gaussian distribution
    :param b: Boundary vector where each realization x should have every component greater than b to be included in the
        output probability
    :return rev_cdf: P(X_i > b_i, for all i) for multivariate gaussian random variable X
    """
    m = mat_r.shape[0]
    assert len(b) == m, "Input sizes dont match"
    counter = 0  # Iteration dec
    rev_cdf = 0

    while counter < 2 ** m:
        # Get the integral Upper limits
        bin_str = count2bin_digits(counter, m)
        up_lim_list = []
        num_1s = 0
        for i, nn in enumerate(bin_str):
            if nn == '0':
                up_lim_list.append(np.inf)
            elif nn == '1':
                up_lim_list.append(b[i])
                num_1s += 1
            else:
                raise Exception("Coding character for integral limit must be binary")

        rev_cdf += np.power(-1, num_1s) * mvn(mean=np.zeros(m), cov=mat_r).cdf(up_lim_list)
        counter += 1
    return rev_cdf


def get_alpha_ij(cov, mu, i, j, mat_a=None, vec_a_i=None, mat_r_i_hat=None):
    """
    Returns the vector alpha_ij in Afonja (1972) Corollary 2.
    -------------
    :param cov: Covariance Matrix of original multivariate gaussian random variables x_i's
    :param mu: Mean of original multivariate gaussian random variables x_i's
    :param i: Index i for calculating R_ij in Afonja (1972)
    :param j: Index j for calculating R_ij in Afonja (1972)

    :param mat_a: Matrix a_ij in Afonja (1972). Computed if not provided (slower)
    :param vec_a_i: Reduced vector a_i in Afonja (1972). Computed if not provided (slower)
    :param mat_r_i_hat: Reduced matrix R_i_hat in Afonja (1972). Computed if not provided (slower)

    :return vec_alpha_ij: Vector a which is used in k-2 dimensional CDF in Afonja (1972)
    """
    assert i != j, "Indexes should be different in R_ij"
    m = cov.shape[0]
    assert i < m and j < m, "Indexes should be within the range"
    assert len(mu) == m, "Input sizes dont match"

    if mat_a is None:
        mat_a = get_a_i(cov, mu)
    if vec_a_i is None:
        vec_a_i = remove_elem_i(mat_a[i, :], i)
    if mat_r_i_hat is None:
        mat_r_i = get_r_i_hat(cov, mu, i)
        mat_r_i_hat = remove_elem_i(mat_r_i, i)

    j_ind = j if j < i else j - 1
    vec_r_ij = remove_elem_i(mat_r_i_hat[:, j_ind], j_ind)  # j_th element removed j_th flow_row of R_i_hat
    a_ij_prime = remove_elem_i(vec_a_i, j_ind)
    numer_alpha_ij = a_ij_prime - mat_a[i, j] * vec_r_ij  # remove index i from mat_a as well
    vec_alpha_ij = numer_alpha_ij / np.sqrt(1 - np.power(vec_r_ij, 2))
    return vec_alpha_ij


def get_r_ij(cov, mu, i, j):
    """
    Returns the correlation matrix R_ij in Afonja (1972) Corollary 2.
    -------------
    :param cov: Covariance Matrix of original multivariate gaussian random variables x_i's
    :param mu: Mean of original multivariate gaussian random variables x_i's
    :param i: Index i for calculating R_ij in Afonja (1972)
    :param j: Index j for calculating R_ij in Afonja (1972)
    :return mat_r_ij: Correlation matrix R_ij whose q,s element is corr(x_i - x_q, x_i - x_s|x_i - x_j)
    """
    assert i != j, "Indexes should be different in R_ij"
    m = cov.shape[0]
    assert i < m and j < m, "Indexes should be within the range"
    assert len(mu) == m, "Input sizes dont match"

    e_i = np.zeros((m, 1))
    e_i[i] = 1  # Unit vector
    mat_k_i = np.ones((m, 1)) @ e_i.T - np.eye(m)

    # Define the difference variable U
    # mu_u = remove_elem_i((mat_k_i @ mu).flatten(), i)
    cov_u = remove_elem_i(mat_k_i @ cov @ mat_k_i.T, i)

    # Conditional mean & covariance
    j_ind = j if j < i else j - 1
    # mu_unobs = remove_elem_i(mu_u, j_ind)
    # mu_obs = mu_u[j_ind]
    cov_unobs = remove_elem_i(cov_u, j_ind)
    cov_obs = cov_u[j_ind, j_ind]
    cov_unobs_obs = remove_elem_i(cov_u[:, j_ind].flatten(), j_ind).reshape((-1, 1))

    # mu_u_cond = mu_unobs + cov_unobs_obs / cov_obs * (u - mu_obs) # u is the conditioned value
    cov_u_cond = cov_unobs - cov_unobs_obs / cov_obs @ cov_unobs_obs.T

    mat_r_ij = cov2corr_coef(cov_u_cond)  # + mu_u_cond.reshape((-1, 1)) @ mu_u_cond.reshape((1, -1))
    # It is understood that normalized covariance matrix is intended with coorelation matrix R in the paper
    return mat_r_ij


def calc_mean_max(cov, mu):
    """
    Given the multivariate gaussian random variable X described by the covariance matrix and mean vector, calculates
    the first moment of the max E[max(X)] based on Afonja (1972).

    :param cov: Covariance Matrix of original multivariate gaussian random variables x_i's
    :param mu: Mean of original multivariate gaussian random variables x_i's
    :return exp_mean: Expected value of the max
    """
    assert scipy.linalg.issymmetric(cov), "Covariance Matrix is not symetric"
    assert is_pos_def(cov), "Covariance matrix is not positive definite"
    m = cov.shape[0]

    # First Term
    mat_a = get_a_i(cov, mu)
    first_term = 0
    for i in range(m):
        alpha_i = remove_elem_i(mat_a[i, :].flatten(), i)
        mat_r_i_hat = get_r_i_hat(cov, mu, i)
        first_term += mu[i] * reverse_mvn_cdf(mat_r_i_hat, alpha_i)

    # Second Term

    # Form Coefficient matrix
    ind = np.ones((m, m)) == 1
    np.fill_diagonal(ind, False)

    sigma_ii = np.dot(np.diag(cov).reshape((m, 1)), np.ones((1, m)))
    sigma_jj = np.dot(np.ones((m, 1)), np.diag(cov).reshape((1, m)))
    var_xi_xj = sigma_ii - 2 * cov + sigma_jj
    coef_mat = np.divide(sigma_ii - cov, np.sqrt(var_xi_xj), where=ind)
    np.fill_diagonal(coef_mat, 0)

    second_term = 0
    for i in range(m):
        mat_r_i_hat = get_r_i_hat(cov, mu, i)
        vec_a_i = remove_elem_i(mat_a[i, :], i)
        for j in range(m):
            if i == j:
                continue
            alpha_ij = get_alpha_ij(cov, mu, i, j, mat_a, vec_a_i, mat_r_i_hat)
            mat_r_ij = get_r_ij(cov, mu, i, j)  # Mean part is not constant !
            second_term += coef_mat[i, j] * mvn(mean=0, cov=1).pdf(mat_a[i, j]) * reverse_mvn_cdf(mat_r_ij, alpha_ij)
    return first_term + second_term


if __name__ == '__main__':

    n = 10
    p = lambda x: 1.5 * np.log(x) / x
    seed = 23
    np.random.seed(seed)

    graph = nx.erdos_renyi_graph(n, p(n), seed=seed, directed=False)
    nx.relabel_nodes(graph, {i: int(i) + 1 for i in graph.nodes}, copy=False)
    print(graph.nodes)

    risks = {i: i for i in graph.nodes}
    print(risks)
    dists = safest_path(graph, 1, risks)
    print(dists)

    # Plotting
    cmap = plt.cm.Reds  # plt.cm.YlOrBr#'Reds'
    node_clr = [dists[node] for node in graph.nodes]
    norm = matplotlib.colors.Normalize(vmin=min(node_clr), vmax=max(node_clr))

    pos = nx.spring_layout(graph, seed=seed)
    risk_pos = [(x - 0.05, y + 0.1) for x, y in pos.values()]
    for val, xy, clr in zip(risks.values(), risk_pos, node_clr):
        plt.annotate(str(val) + ', ', xy)

    for xy, clr in zip(risk_pos, node_clr):
        plt.annotate(str(clr), (xy[0] + 0.07, xy[1]), color=cmap(norm(clr)))
    nx.draw_networkx(graph, pos=pos, cmap=cmap, node_color=node_clr)

    ax = plt.gca()
    cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)  # , pad=0.2)
    plt.show()
