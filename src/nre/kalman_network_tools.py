import datetime
import time
import matplotlib
import networkx as nx
import numpy as np

from copy import deepcopy
from PIL import Image
from filterpy.kalman import KalmanFilter
from matplotlib import pyplot as plt

from .network_connectivity import MIN_SAMPLES, ConnectivityUnit, get_all_entities, single_risk_update
from .time_windowed import get_window


# Kalman Filter Tools
def plot_kalman_res(mat_x, mat_p, str_k='k', fig=None):
    """Plots the Kalman Filter results: Mean states x and Covariance Matrix mat_p"""
    if len(mat_x.shape) <= 1:
        mat_x = mat_x.reshape(len(mat_x), 1)
    if fig is None:
        fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    im = ax.matshow(mat_x, cmap='YlOrBr')
    ax.set_title(r'$\hat{\mathbf{x}}_{' + str_k + '|' + str_k + '}$', fontsize=20)
    plt.xticks([], [])
    fig.colorbar(im)

    ax = fig.add_subplot(1, 2, 2)
    im = ax.matshow(mat_p, cmap='YlGnBu')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_title(r'$\mathbf{P}_{' + str_k + '|' + str_k + '}$', fontsize=20)
    fig.colorbar(im)
    plt.tight_layout()
    return fig


def sum_scores(w, mat_x_kf, mat_p_kf):
    """Returns the sum scores given mean and covariance of risk estimates"""
    return np.squeeze(w.T @ mat_x_kf), np.squeeze(w.T @ mat_p_kf @ w)


def plot_scores(method_scores, method_names):
    """Plots the scores and returns the fig handle. If figure is given plots onto
    the figure"""
    cmap = matplotlib.cm.get_cmap('Spectral')
    colors = [cmap(v) for v in np.linspace(0, 1, num=len(method_scores))]
    fig, ax = plt.subplots()
    for i, score_list in enumerate(method_scores):
        ys = np.array(list(map(lambda x: x[0], score_list)))
        Pys = np.array(list(map(lambda x: x[1], score_list)))

        ax.plot(ys, color=colors[i], label=method_names[i])
        ax.fill_between(np.arange(ys.shape[0]), ys - np.sqrt(Pys), ys + np.sqrt(Pys), color=colors[i], alpha=.1)
    plt.legend(bbox_to_anchor=(1, 0.5))
    plt.ylabel('Scores')
    plt.xlabel('Time Step')
    return fig


# Other Utilities
def graph_evol(all_graphs, forget_factor):
    """Given list of graphs calculated for each window and given forget factor
    calculates the current graph's (past edges and current one) evolution
    """
    assert all_graphs.shape[-1] == all_graphs.shape[-2], 'Graph matrix is not square'
    assert 1 >= forget_factor >= 0, 'Forget factor must be in [0, 1]'
    curr_graph = np.eye(all_graphs.shape[-1])
    curr_graphs = np.zeros(all_graphs.shape)
    for i in range(all_graphs.shape[0]):
        curr_graph = (1 - forget_factor) * curr_graph + forget_factor * np.squeeze(all_graphs[i, :, :])
        curr_graphs[i, :, :] = curr_graph.copy()
    return curr_graphs


# TODO: Simplifies related functions with single_step_update below, add docstring


def graphs_2_risk_scores(all_graphs, all_measurements=None, all_mat_h=None, mat_x_init=None, mat_p_init=None,
                         all_mat_q=None, all_mat_r=None, k_steps=15, relief_factor=0.6, normalize=False,
                         sequential=False, whole_risks=True, w=None, return_cov=False):
    """
    Given graphs, returns the scores according to weights w. Runs kalman filter for k_steps steps

    :param all_graphs: List of all network state graphs
    List of 2D array of size [n_nodes, n_nodes]
    :param all_measurements: List of measurements for each step the graph is calculated
    List of 1D array of size [n_z]
    :param all_mat_h: List of observation matrices that indicate the measured entity
    List of 2D array of size [n_z, n_nodes]
    :param mat_x_init: Initial risk estimate
    1D array of size [n_nodes]
    :param mat_p_init: Initial risk estimate error covariance matrix
    2D array of size [n_nodes, n_nodes]
    :param all_mat_q: All system noise covariance matrices
    List of 2D array of size [n_nodes, n_nodes]
    :param all_mat_r: All measurement noise covariance matrices
    List of 2D array of size [n_z, n_z]

    :param k_steps: Number of Kalman Filter Steps to be used in calculating risk estimates.
    See the paper for more.
    :param relief_factor: Percentage of the risk relieved at each time step for each node.
    See the paper for more.
    :param normalize: If True, normalizes risks at each step
    :param sequential: If True, uses the previous graph's risk estimates as prior risk estimates for current step; else,
        initializes risk prior to uninformative uniform prior for each graph
    :param whole_risks: If True, returns risks of all entities in the network; else, returns the weighted sum according
        to w
    :param w: Weight matrix to be used if returning weighted sum of risks
    :param return_cov: If True, returns the recorded covariance matrix of risk estimates as well

    :return all_means, all_cov (optional):
        mat_x_kf: All calculated Risk Estimates means
        mat_p_kf: All calculated Risk Estimates error covariance matrices
    """
    if w is None:
        w = np.array([])

    assert np.all(graph.shape[-1] == graph.shape[-2] for graph in all_graphs), 'Graph matrix is not square'
    n_nodes = all_graphs[-1].shape[-1]
    n_step = len(all_graphs)

    if all_measurements is not None or all_mat_h is not None:
        assert np.all([len(z) == mat_h.shape[0] for z, mat_h in
                       zip(all_measurements, all_mat_h) if z is not None or mat_h is not None]), \
            'Measurement dimensions mismatch'
        assert n_step == len(all_measurements) and n_step == len(all_mat_h), 'Measurement input mismatch'

    # Initializations
    if mat_x_init is None:
        mat_x_init = np.ones((n_nodes, 1))
        mat_x_init = mat_x_init / np.linalg.norm(mat_x_init)
    if mat_p_init is None:
        mat_p_init = np.eye(n_nodes) / 10 ** 1  # -1
    if all_mat_q is None:
        all_mat_q = [np.eye(n_nodes, n_nodes) / 10 ** 3 for _ in range(n_step)]
    if all_mat_r is None:
        all_mat_r = [np.eye(1, 1) / 10 ** 2 for _ in range(n_step)]

    mat_x_kf = mat_x_init.copy()
    mat_p_kf = mat_p_init.copy()
    all_means = [mat_x_kf]
    all_cov = [mat_p_kf]
    for i in range(n_step):
        # Kalman Filter
        if not sequential:
            mat_x_kf = mat_x_init.copy()
            mat_p_kf = mat_p_init.copy()

        measurement = all_measurements[i] if all_measurements is not None else None
        mat_h = all_mat_h[i] if all_mat_h is not None else None
        mat_x_kf, mat_p_kf = single_risk_update(all_graphs[i], measurement=measurement, mat_h=mat_h,
                                                mat_x_init=mat_x_kf, mat_p_init=mat_p_kf, mat_q=all_mat_q[i],
                                                mat_r=all_mat_r[i], k_steps=k_steps, relief_factor=relief_factor,
                                                normalize=normalize)

        if whole_risks:
            all_means.append(mat_x_kf)
            all_cov.append(mat_p_kf)
        else:
            sum_score, sum_var = sum_scores(w, mat_x_kf, mat_p_kf)
            all_means.append(float(sum_score))
            all_cov.append(mat_p_kf)

    out_vars = all_means
    if return_cov:
        out_vars = out_vars, all_cov
    return out_vars


# TODO: Needs to be renewed
def score_evaluation(df, entity_names, w, conn_param='Num Packets Rec', batch_size=100, time_windows=None,
                     time_scale='sec', window_types=None, domain_types=None, methods=None):
    """
    Calculates the risk scores over time for the constant graph given by df and
    calculation parameters 
    """
    if time_windows is None:
        time_windows = [120]
    if window_types is None:
        window_types = ['time']
    if methods is None:
        methods = ['cov', 'mi']
    if domain_types is None:
        domain_types = ['time']
    method_name_dict = {'cov': 'Corr. Coeff. ', 'mi': 'MI '}
    time_scale_dict = {'sec': 's', 'min': 'min'}
    method_scores = []
    method_names = []
    nm_base = ConnectivityUnit()
    for win_type in window_types:
        for time_window in time_windows:
            nm_base.read_flows(df, conn_size=batch_size, conn_param=conn_param,
                               entity_names=entity_names, window_type=win_type, sync_window_size=time_window,
                               time_scale=time_scale)
            for domain_type in domain_types:
                nm = deepcopy(nm_base)
                if domain_type == 'freq':
                    nm.apply_dft_mag()
                for method in methods:
                    nm.fit_connectivity_model(method=method)
                    if win_type == 'time':
                        windows_str = str(time_window) + time_scale_dict[time_scale] + ' '
                    else:
                        windows_str = str(batch_size) + ' flow '
                    print(type(windows_str))
                    method_name = method_name_dict[
                                      method] + windows_str + win_type + ' window ' + domain_type + ' domain'
                    print(method_name + ':')
                    nm.plot_f()
                    plt.show()

                    n_entity = len(nm.names)
                    n_z = 1
                    mat_x_init = np.ones((n_entity, 1))
                    mat_x_init = mat_x_init / np.linalg.norm(mat_x_init)
                    mat_p_init = np.eye(n_entity) / 10 ** 2.5
                    mat_q = np.eye(n_entity, n_entity) / 10 ** 3
                    mat_r = np.eye(n_z, n_z) / 10 ** 2
                    mat_h = np.zeros((n_z, n_entity))  # Not used

                    mat_f = nm.mat_f.copy()
                    mat_x_kf = mat_x_init.copy()
                    mat_p_kf = mat_p_init.copy()
                    k_steps = 15
                    score_list = [sum_scores(w, mat_x_kf, mat_p_kf)]

                    f = KalmanFilter(dim_x=n_entity, dim_z=n_z)
                    for k in range(k_steps):
                        # Kalman Filter                
                        f.x = mat_x_kf
                        f.F = mat_f
                        f.H = mat_h
                        f.P = mat_p_kf
                        f.Q = mat_q
                        f.R = mat_r

                        f.predict()
                        mat_x_kf = f.x.copy()
                        mat_p_kf = f.P.copy()
                        # Normalization
                        c = np.linalg.norm(mat_x_kf)
                        mat_x_kf = mat_x_kf / c
                        mat_p_kf = mat_p_kf / (c ** 2)

                        score_list.append(sum_scores(w, mat_x_kf, mat_p_kf))
                    method_scores.append(score_list)
                    method_names.append(method_name)
    fig = plot_scores(method_scores, method_names)
    return fig, method_scores, method_names


# NRE Classification Tools
def parse_df_2_state_graphs(df, entity_names=None, method='cov', window_type='time', t_graph=180, date_col=' Timestamp',
                            label_col=' Label', src_id_col=' Source IP', dst_id_col=' Destination IP',
                            time_scale='sec', n_graph=10000, labelling_opt='attacks first', benign_label='BENIGN',
                            skip_idle=True, verbose=True, return_datetimes=False, timeit=False, **kwargs):
    """
    Parses the df into state graphs on entity_names using connection or time windows.

    :param df: Canonical source dataframe that has flow information with timestamp, label and flow features.
        Each row is a flow
    :param entity_names: List of entities that the flows are constrained to
    :param method: ConnectivityUnit method for fitting the graph model to samples
    :param window_type: The windowing type, either 'time' or 'connection'.
    :param t_graph: Time window length that corresponds to a single state graph in time_scale units
    :param date_col: The dataframe df column that holds datetime timestamps of flows
    :param label_col: The dataframe df column that holds the labels of flows
    :param src_id_col: Column name of the flows data DatatFrame that identifies the source entity.
    :param dst_id_col: Column name of the flows data DatatFrame that identifies the destination entity.
    :param time_scale: Time window units. Either 'sec' or 'min'
    :param n_graph: Number connections used to form a single network state graph (only if window_type=='connection')
    :param labelling_opt: Labelling scheme to be used; either 'attacks first' or 'majority'. 'attacks first' labels the
        windows that have a flow anything other than BENIGN as 'attack'. 'majority' option casts a majority vote
        among flows in the window to produce the label
    :param benign_label: Label of the benign, non-malicious, flows
    :param skip_idle: If True, skips over empty windows; if False, results in identity graphs in between
    :param verbose: If True, print the graph number, datetime currently at in df and sample size used to calculate graph
        as running
    :param return_datetimes: If True, returns starting datetimes of each graph calculation as well
    :param timeit: Record and return simulation running times as well
    :param kwargs: ConnectivityUnit.read_flows() keyword arguments

    :return: all_graphs, labels, label_counts, date_times (optional), t_sim (optional)
        all_graphs: Collection of network state graphs
        labels: Labels of the network state according to the labelling scheme
        label_counts: List of labels counts dictionary per each graph
        entity_names: Respective entity names of the graphs
        date_times: (Returned only if return_datetimes==True). List of datetimes used to start computing the graphs
        t_sim: (Returned only if timeit==True) List of running time spent for calculating each graph
    """
    if entity_names is None:
        entity_names = get_all_entities(df, src_id_col=src_id_col, dst_id_col=dst_id_col)
    if timeit:
        start_time = time.time()
        t_sim = []
    n_nodes = len(entity_names)
    all_graphs = np.empty((0, n_nodes, n_nodes))
    labels = []
    label_counts = []
    end_of_df = False
    i = 0
    current_datetime = df.iloc[0][date_col]
    last_datetime = df.iloc[-1][date_col]

    date_times = [current_datetime]
    while end_of_df is False:
        if window_type == 'connection':
            temp_df = df.iloc[i * n_graph: (i + 1) * n_graph, :].copy()
            i += 1
            if i >= df.shape[0] // n_graph:
                end_of_df = True
        else:  # 'time'
            if skip_idle:
                temp_df, _, current_datetime = get_window(current_datetime, df, date_col=date_col, time_window=t_graph,
                                                          time_scale=time_scale, return_next_time=True)
            else:
                temp_df, current_datetime = get_window(current_datetime, df, date_col=date_col, time_window=t_graph,
                                                       time_scale=time_scale)
            date_times.append(current_datetime)
            if current_datetime >= last_datetime:
                end_of_df = True
            if temp_df.empty or len(temp_df.shape) < 2 or temp_df.shape[0] < MIN_SAMPLES:
                temp_graph = np.eye(n_nodes).reshape((1, n_nodes, n_nodes))
                all_graphs = np.concatenate((all_graphs, temp_graph), axis=0)
                labels.append('Empty')
                label_counts.append({})
                if timeit:
                    t_sim.append(time.time() - start_time)
                continue
            i += 1
            print(i) if verbose else None

        cu = ConnectivityUnit()
        delta_datetime = datetime.timedelta(minutes=t_graph) if time_scale == 'min' else datetime.timedelta(
            seconds=t_graph)
        window_last_datetime = temp_df[date_col].iloc[0] + delta_datetime
        cu.read_flows(temp_df, entity_names=entity_names, last_datetime=window_last_datetime, window_type=window_type,
                      date_col=date_col, time_scale=time_scale, **kwargs)
        if verbose:
            print('Current time and samples shape: ', current_datetime, cu.samples.shape)
        cu.fit_connectivity_model(method=method, verbose=False)  # cov

        g = nx.from_numpy_array(cu.mat_f, create_using=nx.DiGraph)
        g = nx.relabel_nodes(g, {entity_names.index(node): node for node in entity_names})
        g.add_weighted_edges_from([(node, node, 1) for node in entity_names])
        temp_graph = np.asarray(nx.to_numpy_array(g, nodelist=entity_names))
        temp_graph = temp_graph.reshape((1, n_nodes, n_nodes))
        all_graphs = np.concatenate((all_graphs, temp_graph), axis=0)

        # Label Counting & Majority Labelling
        values, counts = np.unique(temp_df[label_col].values, return_counts=True)
        temp_counts = {value: count for value, count in zip(values, counts)}
        label_counts.append(temp_counts.copy())

        if labelling_opt == 'attacks first':
            if len(temp_counts) == 1:
                labels.append(str(list(temp_counts.keys())[0]))
            else:
                try:
                    temp_counts.pop(benign_label)
                except KeyError:
                    pass
                ind = np.argmax(list(temp_counts.values()))
                labels.append(str(list(temp_counts.keys())[ind]))
        else:  # Majority labelling
            ind = np.argmax(list(temp_counts.values()))
            labels.append(str(list(temp_counts.keys())[ind]))

        # Running times
        if timeit:
            t_sim.append(time.time() - start_time)
    # Return
    out_vars = [all_graphs, labels, label_counts, entity_names]
    if return_datetimes:
        out_vars.append(date_times)
    if timeit:
        out_vars.append(t_sim)
    return tuple(out_vars)


# TODO: Fix inaccuracies with output variables
def get_risk_mat_from_df(df, forget_factor=0.5, k_steps=1, relief_factor=0.2, return_cov=False, return_datetimes=False,
                         timeit=False, **kwargs):
    """
    Returns the entity risks obtained from windowed ConnectivityUnit

    :param df: Canonical source dataframe that has flow information with timestamp, label and flow features.
        Each row is a flow
    :param forget_factor: Linear interpolation hyperparameter that controls how much previous state graph should
        contribute to the current graph. 1 results in no contribution, 0 results in same state graphs. See the paper for
        more description
    :param k_steps: Number of Kalman Filter Steps to be used in calculating risk estimates. See the paper for more.
    :param relief_factor: Percentage of the risk relieved at each time step for each node (see the paper for more).
    :param return_cov: If True, returns the recorded covariance matrix of risk estimates as well
    :param return_datetimes: If True, returns starting datetimes of each graph calculation as well
    :param timeit: Record and return simulation running times as well
    :param kwargs: ConnectivityUnit.read_flows() and parse_df_2_state_graphs keyword arguments

    :return: risk_mat, labels, label_counts, entity_names, all_cov (optional), date_times (optional), t_sim (optional)
        risk_mat: Collection of entity risk estimates
        labels: Labels of the network state according to the labelling scheme
        label_counts: List of labels counts dictionary per each graph
        entity_names: Ordered list of entities that the flows are constrained to
        all_cov: List of recorded Covariance matrices
        date_times: (Returned only if return_datetimes==True). List of datetimes used to start computing the graphs
        t_sim: (Returned only if timeit==True) List of running time spent for calculating each graph
    """

    out_vars = parse_df_2_state_graphs(df, window_type='time', return_datetimes=return_datetimes,
                                       timeit=timeit, **kwargs)
    out_vars = list(out_vars)

    all_graphs, labels, label_counts, entity_names = out_vars[:4]  # First 4 are always same
    date_times = None
    t_sim = None
    if return_datetimes:
        date_times = out_vars.pop(4)
    if timeit:
        t_sim = out_vars.pop(4)

    all_graphs = graph_evol(all_graphs, forget_factor)
    out_risk = graphs_2_risk_scores(all_graphs, k_steps=k_steps, relief_factor=relief_factor, normalize=True,
                                    sequential=True, whole_risks=True, return_cov=return_cov)
    labels.insert(0, 'Empty')
    label_counts.insert(0, {})
    if return_cov:
        all_risks, all_cov = out_risk
    else:
        all_risks = out_risk
    risk_mat = np.array([list(risks.flatten()) for risks in all_risks])

    out_vars = [risk_mat, labels, label_counts, all_graphs, entity_names]
    if return_cov:
        out_vars.append(all_cov)
    if return_datetimes:
        out_vars.append(date_times)
    if timeit:
        out_vars.append(t_sim)
    return tuple(out_vars)


def stream_df_to_jsonstring(df, forget_factor=0.5, k_steps=1, relief_factor=0.2, return_cov=False, return_datetimes=False,
                         timeit=False, **kwargs):
    """ Returns the string encoding risk estimates and mat_f to be used in dynamic rendering """



# Making GIF
def append_fig_to_frames(fig, frames=None, k=0, save_place='GIF_Images'):
    """Appends figure to frames"""
    if frames is None:
        frames = []
    img_file = save_place + '/Image' + str(k + 1) + '.jpg'
    fig.savefig(img_file, format='jpg')
    im = Image.open(img_file)
    frames.append(im)
    return frames


def save_gif(save_name, frames, dur=400, loop=0, format='GIF'):
    """Saves the gif from frames to save_name"""
    frame_one = frames[0]
    frame_one.save(save_name, format=format, append_images=frames,
                   save_all=True, duration=dur, loop=loop)
