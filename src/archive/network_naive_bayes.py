import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from src.npeet import npeet_entropy_estimators as ee
from ordered_set import OrderedSet
from src.time_windowed import get_window
import scipy.fft
import time
import datetime

# NB functions
# loss_thr = -10  # around here for beta usage
MIN_SAMPLES = 5  # Minimum amount of samples required for MI calculation


class NetworkModel:
    class GaussRvs:
        """Simple Gaussian Model for observations. Each column is assumed to be normal distributed"""

        def __init__(self, samples, node_names):
            m, n = samples.shape
            self.names = node_names
            # Calc the means
            self.mean = np.mean(samples, 0)

            # Calc the covariances
            self.cov = (np.cov(samples.T) * m / (m - 1))

        def __str__(self):
            return 'Mean: ' + str(self.mean) + ',\nCovariance:' + str(self.cov)

        def get_f(self):
            """Returns the graph matrix mat_f of data"""
            variances = np.diag(self.cov)
            mat_f_cov = np.abs(self.cov - np.diag(np.diag(self.cov)))
            var_mult_mat = np.outer(variances, variances)
            return np.divide(mat_f_cov, np.sqrt(var_mult_mat))

        def get_r(self):
            """Returns the diagonal matrix of variances of the RVs"""
            return np.diag(np.diag(self.cov))

    def __init__(self, loss_thr=9999):
        self.LOSS_THR = loss_thr  # loss is 0 if X and Y are independent
        self.CONN_PARAMS = ['Activation', 'Num Packets Sent', 'Num Packets Rec', 'Packet Length',
                            'Flow Duration', 'Flow Speed', 'Response Time', 'Packet Delay', 'Header Length',
                            'Num Active Packets', 'Active Time', 'Idle Time']
        self.base = 2
        self.mi_mapping = lambda x: np.sqrt(1 - np.power(self.base, -(np.abs(x) + x) / 2))

    def read_data(self, df, last_datetime=None, batch_size=100, conn_param='Num Packets Rec', entity_names=None,
                  window_type='time', date_feature=' Timestamp', time_window=20, time_scale='sec'):
        """
        Reads from the data frame. Format: CIC-IDS-2017 type.
        Adds the samples for node RV for conn_param given the connections df
        """
        if entity_names is None:  # Get the node names in df
            node_names = OrderedSet()
            for index, row in df.iterrows():
                node_names.add(row.iloc[1])
                node_names.add(row.iloc[3])
            node_names = list(node_names)
        else:
            node_names = entity_names.copy()
        idle_name = 'Unused'

        node_names.append(idle_name)
        node_ind_dict = {node: node_names.index(node) for node in node_names}
        num_appearances = [0 for node in node_names]

        samples = []

        if window_type == 'time':
            current_datetime = df.iloc[0][date_feature]
            if last_datetime is None:
                last_datetime = df.iloc[-1][date_feature]
            while current_datetime < last_datetime:
                window, current_datetime = get_window(current_datetime, df, time_window, time_scale=time_scale)

                n = 0
                temp = [0 for node in node_names]
                counts = np.zeros(len(node_names))
                for index, row in window.iterrows():
                    s_ind, d_ind = get_s_d_ind(row, node_names, idle_name, node_ind_dict)

                    # Add dec
                    num_appearances[s_ind] += 1
                    num_appearances[d_ind] += 1

                    # Number of instances in a batch for averaging
                    counts[s_ind] += 1
                    counts[d_ind] += 1

                    rv_type = update_temp(temp, row, counts, s_ind, d_ind, conn_param)
                    n += 1

                temp.pop()  # Remove unused entry
                samples.append(temp)

        else:  # 'connection'
            n = 0
            temp = [0 for node in node_names]
            counts = np.zeros(len(node_names))
            for index, row in df.iterrows():
                s_ind, d_ind = get_s_d_ind(row, node_names, idle_name, node_ind_dict)

                # Add dec
                num_appearances[s_ind] += 1
                num_appearances[d_ind] += 1

                # Number of instances in a batch for averaging
                counts[s_ind] += 1
                counts[d_ind] += 1

                rv_type = update_temp(temp, row, counts, s_ind, d_ind, conn_param)

                n += 1
                if n == batch_size:
                    temp.pop()  # Remove unused entry
                    samples.append(temp)
                    temp = [0 for node in node_names]
                    n = 0
                    counts = np.zeros(len(node_names))

        node_names.pop()
        num_appearances.pop()

        self.samples = np.array(samples, dtype='float')  # Wrong Change ASAP
        self.node_names = node_names
        self.rv_type = rv_type
        self.num_appearances = num_appearances

    def apply_dft_mag(self):
        """Applies magnitude of fft to samples on nodes"""
        self.samples = np.abs(scipy.fft.fft(self.samples, axis=0, norm="forward"))

    def discretize(self, apply_log=True, max_M=5, eps=0.0001, count_perc_thr=None):
        """
        Discretizes the samples of Network Model. Chooses size of alphabet as maximal
        size having each bin larger than count_perc_thr percent samples
        """
        if type(count_perc_thr) is not float:
            count_perc_thr = 1 / (4 * max_M)
        X_disc = np.zeros(self.samples.shape).astype(str)
        X = self.samples.copy()
        if apply_log:
            X = np.log(X + eps)

        for i in range(X.shape[1]):
            bottom = min(X[:, i])
            top = max(X[:, i]) * (1 + eps)
            for M in range(max_M, 0, -1):
                bins = np.linspace(bottom, top, M + 1)
                inds = np.digitize(X[:, i], bins).astype(str)
                _, counts = np.unique(inds, return_counts=True)
                if len(counts) < M:
                    continue
                if min(counts) / X.shape[0] >= count_perc_thr:
                    break
            X_disc[:, i] = inds
        return X_disc

    def set_loss_thr(self, LOSS_THR=-0.5):
        "Set the loss threshold for naive bayes graph inference"
        self.LOSS_THR = LOSS_THR  # loss is 0 if X and Y are indep

    def fit_graph_model(self, method='cov', verbose=True, M_di=3, **kwargs):
        """
        Fits the Graph model mat_f and the noise matrix mat_r using the method given
        """
        if method == 'bbn':  # Bayesian Belief Network Search Methods for DiGraph TODO: Keep but add modules
            # Bayesian network from samples
            learnt_model = BayesianNetwork.from_samples(self.samples, state_names=self.node_names)
            G_learnt = get_DAG_from_model(learnt_model)
            # nx.draw(G_learnt, with_labels=True, font_weight='bold')

            A = nx.adjacency_matrix(G_learnt, self.node_names)
            A = A.todense()
            self.F = A

        elif method == 'nb_old':
            mat_f = np.zeros((len(self.node_names), len(self.node_names)))

            for i, node1 in enumerate(self.node_names):
                for j, node2 in enumerate(self.node_names):
                    if j < i:
                        continue
                    elif i == j:
                        mat_f[i, j] = 1
                        continue
                    data1 = [[el] for el in self.samples[:, i]]
                    data2 = [[el] for el in self.samples[:, j]]

                    mi1_2 = ee.mi_old(data1, data2)
                    loss = - self.mi_mapping(mi1_2)
                    if loss < self.LOSS_THR:
                        mat_f[i, j] = -loss
                        mat_f[j, i] = -loss
            self.F = mat_f

        elif method == 'mi':
            if 'mi_mapping' in kwargs.keys():
                self.mi_mapping = kwargs['mi_mapping']
            mat_f = np.zeros((len(self.node_names), len(self.node_names)))

            for i, node1 in enumerate(self.node_names):
                for j, node2 in enumerate(self.node_names):
                    if j < i:
                        continue
                    elif i == j:
                        mat_f[i, j] = 1
                        continue
                    data1 = self.samples[:, i].reshape((-1, 1))
                    data2 = self.samples[:, j].reshape((-1, 1))

                    mi1_2 = ee.mi(data1, data2)
                    loss = - self.mi_mapping(mi1_2)
                    if loss < self.LOSS_THR:
                        mat_f[i, j] = -loss
                        mat_f[j, i] = -loss
            self.F = mat_f

        elif method == 'mi_gauss':
            if 'mi_mapping' in kwargs.keys():
                self.mi_mapping = kwargs['mi_mapping']
            rvs = self.GaussRvs(self.samples, self.node_names)
            mat_f = np.zeros((len(self.node_names), len(self.node_names)))

            for i, node1 in enumerate(self.node_names):
                for j, node2 in enumerate(self.node_names):
                    if i == j:
                        # g.add_edge(node1, node2, weight = 1)
                        mat_f[i, j] = 1
                        continue
                    var1 = rvs.cov[i, i]
                    var2 = rvs.cov[j, j]
                    if var1 == 0 or var2 == 0:
                        rho = 0
                    else:
                        rho = rvs.cov[i, j] / np.sqrt(var1 * var2)

                    # h_1 = 0.5 * np.log(2 * np.pi * np.e * var1)
                    # h_2 = 0.5 * np.log(2 * np.pi * np.e * var2)
                    # h_2_given_1 = h_2 + 0.5 * np.log(1 - rho ** 2)
                    mi1_2 = 0.5 * np.log(1 / (1 - rho ** 2))
                    # h_joint = h_1 + h_2_given_1
                    # loss = (h_2_given_1) / h_2 - 1
                    loss = - self.mi_mapping(mi1_2)

                    if loss < self.LOSS_THR:
                        mat_f[i, j] = -loss
            self.F = mat_f

        elif method == 'di':
            if 'mi_mapping' in kwargs.keys():
                self.mi_mapping = kwargs['mi_mapping']
            mat_f = np.zeros((len(self.node_names), len(self.node_names)))

            for i, node1 in enumerate(self.node_names):
                for j, node2 in enumerate(self.node_names):
                    if i == j:
                        mat_f[i, j] = 1
                        continue
                    data1 = self.samples[:, i].reshape((-1, 1))
                    data2 = self.samples[:, j].reshape((-1, 1))

                    di1_2 = ee.di(data1, data2, M=M_di)
                    loss = - self.mi_mapping(2 * di1_2)
                    if loss < self.LOSS_THR:
                        mat_f[i, j] = -loss
            self.F = mat_f

        else:  # Covariance Method (Assuming gaussian RVs)

            rvs = self.GaussRvs(self.samples, self.node_names)
            A = rvs.cov

            self.adj_mat = A
            mat_f, R = self.get_F_R_from_adjacency_mat(A)
            self.F = mat_f
            self.R = R
        self.check_stability(self.F, verbose=verbose)

    def get_F_R_from_adjacency_mat(self, A):
        """
        Given the adjacency matrix of graph or covariance matrix in gaussian case, returns mat_f and mat_r matrices for
        Kalman filter
        """
        varbs = np.diag(A)
        R = np.diag(np.diag(A))
        A_abs = np.abs(A - np.diag(np.diag(A)))
        var_mult_mat = np.outer(varbs, varbs)
        F = np.divide(A_abs, np.sqrt(var_mult_mat))
        inds = np.where(np.isnan(F))
        F[inds] = 0
        F = F + np.eye(F.shape[0])
        return F, R

    def check_stability(self, A=False, verbose=True):
        "Checks if the matrix A is stable"
        if not np.any(A):
            A = self.F
        K = np.linalg.cond(A)
        # K = 'Not Calculated'
        d = np.linalg.det(A.T * A)
        if verbose == True:
            print('Conditioning number: ', K, '\nDeterminant of mat_f^T*mat_f: ', d)

    def plot_F(self, cbar_font_size=16):
        "Plots the Adjacency matrix of the graph"
        plt.matshow(self.F)
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=cbar_font_size)
        # plt.show()


def get_s_d_ind(row, node_names, idle_name, node_ind_dict):
    "Retunrs the source and destination indices (s_ind, d_ind) from CIC type flow_row"
    src_name = row.iloc[1]
    dst_name = row.iloc[3]
    if src_name in node_names and dst_name in node_names:
        s_ind = node_ind_dict[src_name]
        d_ind = node_ind_dict[dst_name]
    elif dst_name in node_names:
        s_ind = node_ind_dict[idle_name]
        d_ind = node_ind_dict[dst_name]
    elif src_name in node_names:
        s_ind = node_ind_dict[src_name]
        d_ind = node_ind_dict[idle_name]
    else:
        s_ind = node_ind_dict[idle_name]
        d_ind = node_ind_dict[idle_name]
    return s_ind, d_ind


def update_temp(temp, row, counts, s_ind, d_ind, conn_param):
    """Updates the temp samples for a batch with the given indices of the
    new observation"""
    if conn_param == 'Port Num':  # Currently last param. Can be mad most freq
        temp[s_ind] = str(row[2])
        temp[d_ind] = str(row[4])
        rv_type = 'Discrete'
    elif conn_param == 'Protocol':  # Currently last param. Can be mad most freq
        temp[s_ind] = str(row[5])
        temp[d_ind] = str(row[5])
        rv_type = 'Discrete'
    elif conn_param == 'Num Packets Sent':
        temp[s_ind] += int(row[8])
        temp[d_ind] += int(row[9])
        rv_type = 'Discrete'
    elif conn_param == 'Num Packets Rec':
        temp[s_ind] += int(row[9])
        temp[d_ind] += int(row[8])
        rv_type = 'Discrete'
    elif conn_param == 'Packet Length':  # Mean
        temp[s_ind] = (temp[s_ind] / counts[s_ind]) * (counts[s_ind] - 1) + float(row[14]) / counts[s_ind]
        temp[d_ind] = (temp[d_ind] / counts[d_ind]) * (counts[d_ind] - 1) + float(row[18]) / counts[d_ind]
        rv_type = 'Continuous'
    elif conn_param == 'Flow Duration':
        temp[s_ind] += float(row[7])
        temp[d_ind] += float(row[7])
        rv_type = 'Continuous'
    elif conn_param == 'Flow Speed':
        temp[s_ind] = (temp[s_ind] * (counts[s_ind] - 1) + float(row[20])) / counts[s_ind]
        temp[d_ind] = (temp[d_ind] * (counts[d_ind] - 1) + float(row[20])) / counts[d_ind]
        rv_type = 'Continuous'
    elif conn_param == 'Response Time':
        temp[s_ind] = (temp[s_ind] * (counts[s_ind] - 1) + float(row[22])) / counts[s_ind]
        temp[d_ind] = (temp[d_ind] * (counts[d_ind] - 1) + float(row[22])) / counts[d_ind]
        rv_type = 'Continuous'
    elif conn_param == 'Packet Delay':
        temp[s_ind] = (temp[s_ind] * (counts[s_ind] - 1) + float(row[27])) / counts[s_ind]
        temp[d_ind] = (temp[d_ind] * (counts[d_ind] - 1) + float(row[32])) / counts[d_ind]
        rv_type = 'Continuous'
    elif conn_param == 'Header Length':
        temp[s_ind] = (temp[s_ind] * (counts[s_ind] - 1) + float(row[40])) / counts[s_ind]
        temp[d_ind] = (temp[d_ind] * (counts[d_ind] - 1) + float(row[41])) / counts[d_ind]
        rv_type = 'Continuous'
    elif conn_param == 'Num Active Packets':
        temp[s_ind] += int(row[73])
        rv_type = 'Discrete'
    elif conn_param == 'Active Time':
        temp[s_ind] = (temp[s_ind] * (counts[s_ind] - 1) + float(row[75])) / counts[s_ind]
        temp[d_ind] = (temp[d_ind] * (counts[d_ind] - 1) + float(row[75])) / counts[d_ind]
        rv_type = 'Continuous'
    elif conn_param == 'Idle Time':
        temp[s_ind] = (temp[s_ind] * (counts[s_ind] - 1) + float(row[79])) / counts[s_ind]
        temp[d_ind] = (temp[d_ind] * (counts[d_ind] - 1) + float(row[79])) / counts[d_ind]
        rv_type = 'Continuous'
    elif conn_param == 'Activation':
        temp[s_ind] = 1
        temp[d_ind] = 1
        rv_type = 'Discrete'
    else:
        raise Exception(conn_param + " as connection parameter is not defined.")
    return rv_type


def sigmoid(x):
    "Sigmoid function"
    return 1 / (1 + np.exp(-x))


def calc_cond_loss(joint_prob):
    """
    Given Joint probability distribution calculates the custom loss for heuristic
    A causes B identification

    joint_prob:[[..., ...
                          ]]
    """
    alpha = 1 / 0.1
    beta = 2
    score = np.log(joint_prob[1][1]) - 2 * np.log(joint_prob[0][1]) + np.log(joint_prob[0][0]) - beta * np.log(
        np.abs(joint_prob[1][1] - joint_prob[1][0]))
    # score = np.log(joint_prob[1][1]) - 2*np.log(joint_prob[0][1]) + 2*np.log(joint_prob[0][0]) - np.log(joint_prob[1][0]) - alpha*np.abs(joint_prob[1][1] - joint_prob[1][0])
    return - score


def normalize_2joint(counts):
    "Normalizes the counts"
    joint_prob = []
    tot_count = 0
    for row in counts:
        for entry in row:
            tot_count += entry

    for row in counts:
        temp = []
        for entry in row:
            temp.append(entry / tot_count)
        joint_prob.append(temp.copy())

    return joint_prob


def get_binary_counts(samples, i, j):
    "Returns joint counts (!A, A vs !B, B) table from samples. A and B appear at i and j indices"
    counts = [[0, 0], [0, 0]]
    for obs in samples:
        if obs[i] == '0':
            if obs[j] == '0':
                counts[0][0] += 1
            else:
                counts[0][1] += 1
        else:
            if obs[j] == '0':
                counts[1][0] += 1
            else:
                counts[1][1] += 1
    return counts


def get_counts(samples, i, j):
    "Returns joint counts (!A, A vs !B, B) table from samples. A and B appear at i and j indices"
    counts = [[0, 0], [0, 0]]
    for obs in samples:
        if obs[i] == 0:
            if obs[j] == 0:
                counts[0][0] += 1
            else:
                counts[0][1] += 1
        else:
            if obs[j] == 0:
                counts[1][0] += 1
            else:
                counts[1][1] += 1
    return counts


# old
def get_nb_graph(samples, node_names, LOSS_THR):
    "Calculates the graph from samples using naive bayes (assuming each edge is indep.)"

    nb_graph = nx.DiGraph()
    nb_graph.add_nodes_from(node_names)

    for i, node1 in enumerate(node_names):
        for j, node2 in enumerate(node_names):
            if i == j:
                continue
            counts = get_binary_counts(samples, i, j)
            # print(counts)
            joint_prob = normalize_2joint(counts)
            loss = calc_cond_loss(joint_prob)
            # print(i, j, loss)
            if loss < LOSS_THR:
                nb_graph.add_edge(node1, node2)
    return nb_graph


"""
# ENTROPY ESTIMATION FUNCTIONS
def avgdigamma(points,dvec):
  #This part finds number of neighbors in some radius in the marginal space
  #returns expectation value of <psi(nx)>
  N = len(points)
  tree = KDTree(points)
  avg = 0.
  for i in range(N):
    dist = dvec[i]
    #subtlety, we don't include the boundary point, 
    #but we are implicitly adding 1 to kraskov def bc center point is included
    num_points = len(tree.query_radius(points[i],dist-1e-6,p=np.inf)[0])
    avg += digamma(num_points)/N
  return avg

def avgdigamma2(points, tree, dvec):
    #This part finds number of neighbors in some radius in the marginal space
    #returns expectation value of <psi(nx)>
    N = len(points)
    #subtlety, we don't include the boundary point, 
    #but we are implicitly adding 1 to kraskov def bc center point is included
    ball_points = [tree.query_radius(points[i], dvec[i]-1e-6, p=np.inf)[0] for i in range(N)]
    num_pointss = list(map(len, ball_points))
    return np.sum(digamma(num_pointss))/N

def fast_mi(x, y, k=3, base=2):
    # Mutual information of x and y
    #x,y should be 2d arrays, e.g. x = np.array([[1.3],[3.7],[5.1],[2.4]])
    #if x is a one-dimensional scalar and we have four samples
    assert x.shape[0] == y.shape[0], "Lists should have same length"
    N, d = x.shape
    assert k <= x.shape[0] - 1, "Set k smaller than num. samples - 1"
    intens = 1e-10 #small noise to break degeneracy, see doc.
    x += intens*nr.rand(N, d)
    y += intens*nr.rand(N, d)
    points = np.concatenate((x, y), axis= 1)
    #Find nearest neighbors in joint space, p=inf means max-norm
    tree = KDTree(points)
    dvec = [tree.query(point,k+1,p= np.inf)[0][0, k] for point in points]
    a,b,c,d = avgdigamma(x,dvec), avgdigamma(y,dvec), digamma(k), digamma(N) 
    return (-a-b+c+d)/np.log(base)

def fast_mi2(x, y, k=3, base=2):
    # Mutual information of x and y
    #x,y should be 2d arrays, e.g. x = np.array([[1.3],[3.7],[5.1],[2.4]])
    #if x is a one-dimensional scalar and we have four samples

    assert x.shape[0] == y.shape[0], "Lists should have same length"
    N, d = x.shape
    assert k <= x.shape[0] - 1, "Set k smaller than num. samples - 1"
    intens = 1e-10 #small noise to break degeneracy, see doc.
    x += intens*nr.rand(N, d)
    y += intens*nr.rand(N, d)
    points = np.concatenate((x, y), axis= 1)
    #Find nearest neighbors in joint space, p=inf means max-norm
    tree = KDTree(points)
    dvec = tree.query(points, k+1, p= np.inf)[0][:,k]
    x_tree, y_tree = KDTree(x), KDTree(y)
    a,b,c,d = avgdigamma2(x, x_tree, dvec), avgdigamma2(y, y_tree, dvec), digamma(k), digamma(N) 
    return (-a-b+c+d)/np.log(base)

def get_nb_F_from_samples(samples, sub_net_names, loss_thr):
    #Calculates the graph matrix mat_f using nb method from the samples
    mat_f = np.zeros((len(sub_net_names), len(sub_net_names)))
    for i, node1 in enumerate(sub_net_names):
        for j, node2 in enumerate(sub_net_names):
            if j < i:
                continue
            elif i == j:
                mat_f[i, j] = 1
                continue
            x1 = np.ascontiguousarray(samples[:, i])
            data1 = np.reshape(x1, (-1, 1))
            x2 = np.ascontiguousarray(samples[:, j])
            data2 = np.reshape(x2, (-1, 1))

            mi1_2 = fast_mi(data1, data2)
            loss = - np.tanh(mi1_2)
            if loss < loss_thr:
                mat_f[i, j] = -loss
                mat_f[j, i] = -loss
    return mat_f
"""


# Network and Plotting TODO: Move em out

def plot_mat(A):
    "Custom plotting function for matrices. PLots colormap and displays numbers "
    fig, ax = plt.subplots()
    plot = ax.matshow(A, cmap=plt.cm.Blues)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            c = A[j, i]
            ax.text(i, j, "{:.1f}".format(c), va='center', ha='center')
    plt.colorbar(plot)


def parse_df_2_state_graphs(df, entity_names, window_type='time', method='cov', conn_param='Num Packets Rec',
                            labelling_opt='majority', n_graph=10000, t_graph=5000, date_feature=' Timestamp',
                            time_scale='sec', time_window=20, skip_idle=True, verbose=True, return_datetimes=False,
                            timeit=False):
    """
    Parses the df into state graphs on entity_names using size N connection
    windows.
    skip_idle: True if you want to skip over empty windows, False if wanna add identity graph
    """
    if timeit:
        start = time.time()
        t_sim = []
    n_nodes = len(entity_names)
    all_graphs = np.empty((0, n_nodes, n_nodes))
    # all_graphs = np.zeros((df.shape[0]//N, n_nodes, n_nodes))
    labels = []
    label_counts = []
    end_of_df = False
    i = 0
    current_datetime = df.iloc[0][date_feature]
    last_datetime = df.iloc[-1][date_feature]

    date_times = [current_datetime]
    while end_of_df is False:
        if window_type == 'connection':
            temp_df = df.iloc[i * n_graph: (i + 1) * n_graph, :].copy()
            i += 1
            if i >= df.shape[0] // n_graph:
                end_of_df = True
        else:  # 'time'
            if skip_idle:
                temp_df, _, current_datetime = get_window(current_datetime,
                                                          df, time_window=t_graph, time_scale=time_scale, return_next_time=True)
            else:
                temp_df, current_datetime = get_window(current_datetime,
                                                       df, time_window=t_graph, time_scale=time_scale)
            date_times.append(current_datetime)
            if current_datetime >= last_datetime:
                end_of_df = True
            if temp_df.empty or len(temp_df.shape) < 2 or temp_df.shape[0] < MIN_SAMPLES:
                temp_graph = np.eye(n_nodes).reshape((1, n_nodes, n_nodes))
                all_graphs = np.concatenate((all_graphs, temp_graph), axis=0)
                labels.append('Empty')
                label_counts.append({})
                if timeit:
                    t_sim.append(time.time() - start)
                continue
            i += 1
            print(i) if verbose else None

        nm = NetworkModel()
        delta_datetime = datetime.timedelta(minutes=t_graph) if time_scale == 'min' else datetime.timedelta(
            seconds=t_graph)
        window_last_datetime = temp_df[' Timestamp'].iloc[0] + delta_datetime
        nm.read_data(temp_df, window_last_datetime, window_type=window_type, conn_param=conn_param,
                     entity_names=entity_names, time_window=time_window, time_scale=time_scale)  # 5
        if verbose:
            print('Current time and samples shape: ', current_datetime, nm.samples.shape)
        nm.fit_graph_model(method=method, verbose=False)  # cov

        g = nx.from_numpy_matrix(nm.F, create_using=nx.DiGraph)
        g = nx.relabel_nodes(g, {entity_names.index(node): node for node in entity_names})
        g.add_weighted_edges_from([(node, node, 1) for node in entity_names])
        temp_graph = np.asarray(nx.to_numpy_matrix(g, nodelist=entity_names))
        temp_graph = temp_graph.reshape((1, n_nodes, n_nodes))
        all_graphs = np.concatenate((all_graphs, temp_graph), axis=0)

        # Label Counting & Majority Labelling
        values, counts = np.unique(temp_df[' Label'].values, return_counts=True)
        temp_counts = {value: count for value, count in zip(values, counts)}
        label_counts.append(temp_counts.copy())

        if labelling_opt == 'attacks first':
            if len(temp_counts) == 1:
                labels.append(list(temp_counts.keys())[0])
            else:
                temp_counts.pop('BENIGN')
                ind = np.argmax(temp_counts.values)
                labels.append(list(temp_counts.keys())[ind])
        else:  # Majority labelling
            ind = np.argmax(temp_counts.values)
            labels.append(list(temp_counts.keys())[ind])
        # Running times
        if timeit:
            t_sim.append(time.time() - start)
    # Return
    out_vars = [all_graphs, labels, label_counts]
    if return_datetimes:
        out_vars.append(date_times)
    if timeit:
        out_vars.append(t_sim)
    return tuple(out_vars)
