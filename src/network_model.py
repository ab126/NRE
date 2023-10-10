import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from .npeet import npeet_entropy_estimators as ee
from ordered_set import OrderedSet
from .time_windowed import get_window
import scipy.fft
from pomegranate.bayesian_network import BayesianNetwork
from src.archive.bbn_functions import get_DAG_from_model

MIN_SAMPLES = 5  # Minimum number of samples required for MI calculation

# TODO: Cant check CICFlowFeaatures, the csv file is gone... Unsure about Response Time, Packet Delay,
#  Num Active Packets, Active Time. Other than that looks good
cic_conn_param_specs = {
    'Port Num': {'method': 'last', 'src_feature_col': ' Source Port', 'dst_feature_col': ' Destination Port'},
    'Protocol': {'method': 'last', 'src_feature_col': ' Protocol', 'dst_feature_col': ' Protocol'},
    'Num Packets Sent': {'method': 'total', 'src_feature_col': ' Total Fwd Packets',
                         'dst_feature_col': ' Total Backward Packets'},
    'Num Packets Received': {'method': 'total', 'src_feature_col': ' Total Backward Packets',
                             'dst_feature_col': ' Total Fwd Packets'},
    'Packet Length': {'method': 'average', 'src_feature_col': ' Fwd Packet Length Mean',
                      'dst_feature_col': ' Bwd Packet Length Mean'},
    'Flow Duration': {'method': 'total', 'src_feature_col': ' Flow Duration', 'dst_feature_col': ' Flow Duration'},
    'Flow Speed': {'method': 'average', 'src_feature_col': 'Flow Bytes/s', 'dst_feature_col': 'Flow Bytes/s'},
    'Response Time': {'method': 'average', 'src_feature_col': ' Flow IAT Mean', 'dst_feature_col': ' Flow IAT Mean'},
    'Packet Delay': {'method': 'average', 'src_feature_col': ' Fwd IAT Mean', 'dst_feature_col': ' Bwd IAT Mean'},
    'Header Length': {'method': 'average', 'src_feature_col': ' Fwd Header Length',
                      'dst_feature_col': ' Bwd Header Length'},
    'Num Active Packets': {'method': 'total', 'src_feature_col': ' act_data_pkt_fwd',
                           'dst_feature_col': ' act_data_pkt_fwd'},
    'Active Time': {'method': 'average', 'src_feature_col': 'Active Mean', 'dst_feature_col': 'Active Mean'},
    'Idle Time': {'method': 'average', 'src_feature_col': 'Idle Mean', 'dst_feature_col': 'Idle Mean'},
    'Activation': {'method': 'activation'}
}


# TODO: Add measurement methods
class NetworkModel:
    """
    Base Network Risk Estimation (NRE) Model.

    """

    def __init__(self, loss_thr=9999):

        self.loss_thr = loss_thr  # loss is 0 if X and Y are independent
        self.conn_params = ['Activation', 'Num Packets Sent', 'Num Packets Rec', 'Packet Length',
                            'Flow Duration', 'Flow Speed', 'Response Time', 'Packet Delay', 'Header Length',
                            'Num Active Packets', 'Active Time', 'Idle Time']
        self.conn_param_specs = cic_conn_param_specs
        self.samples = np.array([[]])
        self.names = []
        self.num_appearances = []

        self.base = 2
        self.mi_mapping = lambda x: np.sqrt(1 - np.power(self.base, -(np.abs(x) + x) / 2))  # np.tanh
        self.m_di = 3  # (applies only for method='di') Memory length for calculating Directed Information

        self.F = np.array([[]])
        self.R = np.array([[]])

    def read_flows(self, df, entity_names=None, window_type='time', date_col=' Timestamp', last_datetime=None,
                   sync_window_size=1.2, time_scale='sec', conn_size=50, src_id_col=' Source IP',
                   dst_id_col=' Destination IP', conn_param='Num Packets Received',
                   src_feature_col=' Total Fwd Packets',
                   dst_feature_col=' Total Backward Packets', method='average'):
        """
        Read flows from the DataFrame, computes the samples for the respective connection parameter
        ----------
        :param df: Canonical DataFrame where each row is a flow and columns include timestamp, entity identifiers, and
                   flow features.
        :param entity_names: The List of entity names that the reading and sample computation is limited to. If None,
                             computes for every entity.
        :param window_type: Windowing type, either 'time' or 'connection'.
        :param date_col: Column name of DataFrame that corresponds to the timestamps of flows. Datetime object
                              preferred.
        :param last_datetime: Cutoff timestamp where further flows are ignored. If None, all the flows are added
                              computed.
        :param sync_window_size: Time window size for synchronizing flows.
        :param time_scale: Timescale used for synchronization windows. Must be 'sec' or 'min'.
        :param conn_size: Number of connections used for synchronizing flows.
        :param src_id_col: Column name of the flows data DatatFrame that identifies the source entity.
        :param dst_id_col: Column name of the flows data DatatFrame that identifies the destination entity.
        :param conn_param: Predefined connection parameter that the samples are calculated for. Overwrites method,
                           src_feature_col and dst_feature_col.
        :param src_feature_col: Column name of the flows data DatatFrame that indicates source entity flow feature.
        :param dst_feature_col: Column name of the flows data DatatFrame that indicates destination entity flow feature.
        :param method: Flow Aggregation Method used for computing samples. Must be in ['average', 'total', 'last', 'activation'].
                'average': Average over flows of entities
                'total': Sum over flows of entities
                'last': Last flow of entities
                'activation': Binary values for entities indicating if an entity was active during the time window (1)
                              or not (0).
        :return: None
        """
        if entity_names is None:  # Get all the names in the network
            sub_net_names = get_all_entities(df, src_id_col=src_id_col, dst_id_col=dst_id_col)
        else:
            sub_net_names = entity_names.copy()
        idle_name = 'Unused'

        sub_net_names.append(idle_name)
        node_ind_dict = {node: sub_net_names.index(node) for node in sub_net_names}
        num_appearances = [0 for _ in sub_net_names]
        samples = []
        if window_type == 'time':
            current_datetime = df.iloc[0][date_col]
            if last_datetime is None:
                last_datetime = df.iloc[-1][date_col]
            while current_datetime < last_datetime:
                window, current_datetime = get_window(current_datetime, df, date_col=date_col,
                                                      time_window=sync_window_size, time_scale=time_scale)

                temp = [0 for _ in sub_net_names]
                counts = np.zeros(len(sub_net_names))
                for row in window.to_dict('records'):  # Faster loop through rows
                    s_ind, d_ind = _get_s_d_ind(row, sub_net_names, node_ind_dict, idle_name=idle_name,
                                                src_id_col=src_id_col, dst_id_col=dst_id_col)

                    if conn_param is None:
                        _update_sample(temp, row, counts, s_ind, d_ind, src_feature_col, dst_feature_col, method=method)
                    else:
                        kwargs = self.conn_param_specs[conn_param]
                        _update_sample(temp, row, counts, s_ind, d_ind, **kwargs)

                    # Total Samples Count
                    num_appearances[s_ind] += 1
                    num_appearances[d_ind] += 1

                temp.pop()  # Remove unused entry
                samples.append(temp)

        else:  # 'connection' 
            n = 0
            temp = [0 for _ in sub_net_names]
            counts = np.zeros(len(sub_net_names))
            for index, row in df.iterrows():
                s_ind, d_ind = _get_s_d_ind(row, sub_net_names, node_ind_dict, idle_name=idle_name,
                                            src_id_col=src_id_col, dst_id_col=dst_id_col)

                if conn_param is None:
                    pass
                    _update_sample(temp, row, counts, s_ind, d_ind, src_feature_col, dst_feature_col, method=method)
                else:
                    kwargs = self.conn_param_specs[conn_param]
                    _update_sample(temp, row, counts, s_ind, d_ind, **kwargs)

                # Total Samples Count
                num_appearances[s_ind] += 1
                num_appearances[d_ind] += 1

                n += 1
                if n == conn_size:
                    temp.pop()  # Remove unused entry
                    samples.append(temp)
                    temp = [0 for _ in sub_net_names]
                    n = 0
                    counts = np.zeros(len(sub_net_names))

        sub_net_names.pop()
        num_appearances.pop()

        self.samples = np.array(samples, dtype='float')
        self.names = sub_net_names
        self.num_appearances = num_appearances

    def read_stream(self):
        """ A function to read stream of data"""
        pass

    def apply_dft_mag(self):
        """Applies magnitude of fft to samples on nodes"""
        self.samples = np.abs(scipy.fft.fft(self.samples, axis=0, norm="forward"))

    def discretize(self, apply_log=True, max_m=5, eps=0.0001, count_perc_thr=None):
        """
        Discretizes the samples of Network Model. Chooses size of alphabet as maximal
        size having each bin larger than count_perc_thr percent samples
        """
        if type(count_perc_thr) is not float:
            count_perc_thr = 1 / (4 * max_m)
        mat_x_disc = np.zeros(self.samples.shape).astype(str)
        mat_x = self.samples.copy()
        if apply_log:
            mat_x = np.log(mat_x + eps)

        for i in range(mat_x.shape[1]):
            bottom = min(mat_x[:, i])
            top = max(mat_x[:, i]) * (1 + eps)
            for M in range(max_m, 0, -1):
                bins = np.linspace(bottom, top, M + 1)
                indices = np.digitize(mat_x[:, i], bins).astype(str)
                _, counts = np.unique(indices, return_counts=True)
                if len(counts) < M:
                    continue
                if min(counts) / mat_x.shape[0] >= count_perc_thr:
                    break
            mat_x_disc[:, i] = indices
        return mat_x_disc

    def set_loss_thr(self, loss_thr=-0.5):
        """Set the loss threshold for naive bayes graph inference"""
        self.loss_thr = loss_thr  # loss is 0 if X and Y are independent

    def fit_graph_model(self, method='cov', verbose=True):
        """
        Fits the Graph model mat_f and the noise matrix mat_r using the method given
        -------------
        :param method: Method for fitting the graph model to samples
                'bbn': Bayesian Belief Network Search Methods for DiGraph
                'nb_old' : # Old Naive Bayes Method
                'mi' : Mutual Information Method where every edge weight is the MI between respective samples of
                    entities
                'mi_gauss' : Mutual Information Method with gaussian/normal assumption on samples
                'di' : Directed Information Method where every edge weight is the DI between respective samples of
                    entities, with given memory length
                'cov' : (default) Correlation Coefficient Method where every edge weight is the Corr. Coeff. between
                    respective samples of entities
        :param verbose: If True, prints graph matrix, mat_f, stability measures
        :return : None
        """
        if method == 'bbn':
            # Bayesian network from samples
            learnt_model = BayesianNetwork.from_samples(self.samples, state_names=self.names)
            g_learnt = get_DAG_from_model(learnt_model)
            # nx.draw(G_learnt, with_labels=True, font_weight='bold')

            sample_cov = nx.adjacency_matrix(g_learnt, self.names)
            sample_cov = sample_cov.todense()
            self.F = sample_cov

        elif method == 'nb_old':
            mat_f = np.zeros((len(self.names), len(self.names)))

            for i, node1 in enumerate(self.names):
                for j, node2 in enumerate(self.names):
                    if j < i:
                        continue
                    elif i == j:
                        mat_f[i, j] = 1
                        continue
                    data1 = [[el] for el in self.samples[:, i]]
                    data2 = [[el] for el in self.samples[:, j]]

                    mi1_2 = ee.mi_old(data1, data2)
                    loss = - self.mi_mapping(mi1_2)
                    if loss < self.loss_thr:
                        mat_f[i, j] = -loss
                        mat_f[j, i] = -loss
            self.F = mat_f

        elif method == 'mi':
            mat_f = np.zeros((len(self.names), len(self.names)))

            for i, node1 in enumerate(self.names):
                for j, node2 in enumerate(self.names):
                    if j < i:
                        continue
                    elif i == j:
                        mat_f[i, j] = 1
                        continue
                    data1 = self.samples[:, i].reshape((-1, 1))
                    data2 = self.samples[:, j].reshape((-1, 1))

                    mi1_2 = ee.mi(data1, data2)
                    loss = - self.mi_mapping(mi1_2)
                    if loss < self.loss_thr:
                        mat_f[i, j] = -loss
                        mat_f[j, i] = -loss
            self.F = mat_f

        elif method == 'mi_gauss':
            _, sample_cov = fit_mvn_to_samples(self.samples)
            mat_f = np.zeros(sample_cov.shape)

            for i, node1 in enumerate(self.names):
                for j, node2 in enumerate(self.names):
                    if i == j:
                        # g.add_edge(node1, node2, weight = 1)
                        mat_f[i, j] = 1
                        continue
                    var1 = sample_cov[i, i]
                    var2 = sample_cov[j, j]
                    if var1 == 0 or var2 == 0:
                        rho = 0
                    else:
                        rho = sample_cov[i, j] / np.sqrt(var1 * var2)

                    # h_1 = 0.5 * np.log(2 * np.pi * np.e * var1)
                    # h_2 = 0.5 * np.log(2 * np.pi * np.e * var2)
                    # h_2_given_1 = h_2 + 0.5 * np.log(1 - rho ** 2)
                    # h_joint = h_1 + h_2_given_1
                    # loss = (h_2_given_1) / h_2 - 1
                    mi1_2 = 0.5 * np.log(1 / (1 - rho ** 2))
                    loss = - self.mi_mapping(mi1_2)

                    if loss < self.loss_thr:
                        mat_f[i, j] = -loss
            self.F = mat_f

        elif method == 'di':
            mat_f = np.zeros((len(self.names), len(self.names)))

            for i, node1 in enumerate(self.names):
                for j, node2 in enumerate(self.names):
                    if i == j:
                        mat_f[i, j] = 1
                        continue
                    data1 = self.samples[:, i].reshape((-1, 1))
                    data2 = self.samples[:, j].reshape((-1, 1))

                    di1_2 = ee.di(data1, data2, M=self.m_di)
                    loss = - self.mi_mapping(2 * di1_2)
                    if loss < self.loss_thr:
                        mat_f[i, j] = -loss
            self.F = mat_f

        else:  # Covariance Method (Assuming gaussian RVs)
            _, sample_cov = fit_mvn_to_samples(self.samples)

            mat_f, mat_r = get_mat_f_q_from_covariance(sample_cov)
            self.F = mat_f
            self.R = mat_r
        self.check_stability(self.F, verbose=verbose)

    def check_stability(self, mat_a=False, verbose=True):
        """Check if the matrix mat_a is stable"""
        if not np.any(mat_a):
            mat_a = self.F
        cond_num = np.linalg.cond(mat_a)
        d = np.linalg.det(mat_a.T * mat_a)
        if verbose is True:
            print('Conditioning number: ', cond_num, '\nDeterminant of F^T*F: ', d)

    def plot_f(self, labels=False, cbar_font_size=16):
        """Plots the Adjacency matrix of the graph"""
        plt.matshow(self.F)
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=cbar_font_size)
        if labels:
            plt.title(r'$\mathbf{F}^{(t)}$', fontsize=20)
            plt.xlabel('Entity Index', weight='bold', fontsize=16)
            plt.ylabel('Entity Index', weight='bold', fontsize=16)
            ax = plt.gca()
            ax.xaxis.set_ticks_position('bottom')
        plt.show()


def get_all_entities(df, src_id_col=' Source IP', dst_id_col=' Destination IP'):
    """
    Given a flow DataFrame df returns all the entities that are present in it

    :param df: Canonical DataFrame where each row is a flow and columns include timestamp, entity identifiers and
                   flow features.
    :param src_id_col: Column name of the flows data DatatFrame that identifies the source entity.
    :param dst_id_col: Column name of the flows data DatatFrame that identifies the destination entity.
    :return entity_names: List of entities that are present in the flow DataFrame df
    """
    entity_names = OrderedSet()
    for row in df.to_dict('records'):  # Faster loop through rows
        entity_names.add(row[src_id_col])
        entity_names.add(row[dst_id_col])
    entity_names = list(entity_names)
    return entity_names


def fit_mvn_to_samples(samples):
    """
    Calculates the unbiased estimate of mean and covariance matrix of multivariate normal/gaussian fit to the samples
    -----------
    :param samples: nxn array of samples.
    :return mean, cov: Mean vector and Covariance Matrix of the multivariate normal fit.
    """
    mean = np.mean(samples, 0)
    cov = np.cov(samples.T)
    return mean, cov


def _get_s_d_ind(flow_row, sub_net_names, node_ind_dict, idle_name='Unused', src_id_col=' Source IP',
                 dst_id_col=' Destination IP'):
    """
    Get the source and destination indices according to index dictionary.
    --------
    :param flow_row: DataFrame or dictionary corresponding to the flow
    :param sub_net_names: List of entity names in the subnetwork of interest
    :param node_ind_dict: Index dictionary of node:index
    :param idle_name: Name used if entity is not in the subnetwork of interest
    :param src_id_col: Column name of the flows data DatatFrame that identifies the source entity
    :param dst_id_col: Column name of the flows data DatatFrame that identifies the destination entity
    :return s_ind, d_ind: Source and Destination indices corresponding to the source and destination of the flow
    """

    src_name = flow_row[src_id_col]
    dst_name = flow_row[dst_id_col]
    if src_name in sub_net_names and dst_name in sub_net_names:
        s_ind = node_ind_dict[src_name]
        d_ind = node_ind_dict[dst_name]
    elif dst_name in sub_net_names:
        s_ind = node_ind_dict[idle_name]
        d_ind = node_ind_dict[dst_name]
    elif src_name in sub_net_names:
        s_ind = node_ind_dict[src_name]
        d_ind = node_ind_dict[idle_name]
    else:
        s_ind = node_ind_dict[idle_name]
        d_ind = node_ind_dict[idle_name]
    return s_ind, d_ind


def _update_sample(window_sample, flow_row, counts, s_ind, d_ind, src_feature_col=None, dst_feature_col=None,
                   method='average'):
    """
    Updates the window sample for a synchronization window with the given indices of the new observation
    ---------------
    :param window_sample: Samples of entities for the synchronization window
    :param flow_row: DataFrame or dictionary corresponding to the flow
    :param counts: Total count dictionary of flows for each entity
    :param s_ind: Index in window_sample corresponding to the source entity
    :param d_ind: Index in window_sample corresponding to the destination entity
    :param src_feature_col: Column name of the flows data DatatFrame that indicates source entity flow feature.
    :param dst_feature_col: Column name of the flows data DatatFrame that indicates destination entity flow feature.
    :param method: Flow Aggregation Method used for computing samples. Must be in ['average', 'total', 'last',
                   'activation'].
            'average': Average over flows of entities
            'total': Sum over flows of entities
            'last': Last flow of entities
            'activation': Binary values for entities indicating if an entity was active during the time window (1)
                          or not (0).
    :return: None
    """
    if src_feature_col is None or dst_feature_col is None:
        assert method == 'activation', "'src_feature_col' or 'dst_feature_col' is not provided."

    if method == 'average':
        window_sample[s_ind] = (window_sample[s_ind] * counts[s_ind] + float(flow_row[src_feature_col])) / (
                counts[s_ind] + 1)
        window_sample[d_ind] = (window_sample[d_ind] * counts[d_ind] + float(flow_row[dst_feature_col])) / (
                counts[d_ind] + 1)
        # Number of instances in a window
        counts[s_ind] += 1
        counts[d_ind] += 1
    elif method == 'total':
        window_sample[s_ind] += float(flow_row[src_feature_col])
        window_sample[d_ind] += float(flow_row[dst_feature_col])
    elif method == 'last':
        window_sample[s_ind] = flow_row[src_feature_col]
        window_sample[d_ind] = flow_row[dst_feature_col]
    elif method == 'activation':
        window_sample[s_ind] = 1
        window_sample[d_ind] = 1
    else:
        raise Exception("type must be in ['average', 'total', 'last', 'activation']")


def get_mat_f_q_from_covariance(cov):
    """
    Given the covariance matrix of entities, returns mat_f and mat_q matrices for Kalman filter

    :param cov: Covariance matrix of samples of entities
    :return mat_f, mat_q:
        mat_f: Linear System Model in Kalman Filter formulation. Entries are positive or zeros if respective variance is
         zero
        mat_q: System Noise Covariance matrix. Taken as respective variances in covariance matrix
    """
    variances = np.diag(cov)
    mat_q = np.diag(np.diag(cov))
    var_mult_mat = np.sqrt(np.outer(variances, variances))
    mat_f = np.eye(cov.shape[0])
    np.divide(np.abs(cov), var_mult_mat, out=mat_f, where=var_mult_mat != 0)
    return mat_f, mat_q


def _sigmoid(x):
    """Sigmoid function"""
    return 1 / (1 + np.exp(-x))
