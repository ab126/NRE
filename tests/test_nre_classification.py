import numpy as np
import pandas as pd
import seaborn as sns
import copy
import numpy.random as nr

from matplotlib import pyplot as plt

from src.analyze_cic_ids import nre_classification

from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from src.classification_tools import plot_roc_curves
from src.network_model import get_mat_f_q_from_covariance
from src.preprocess import preprocess_df
from test_network_model_graph import generate_random_mvn_model, samples2flows


def get_n_dim_rotation_matrix(n, angle, flip=True):
    """Generate an n dimensional rotation (real unitary) matrix. Each plane is rotated by +/- angle degrees """
    assert n >= 2, "Number of dimensions must be at least 2"
    angles = [[angle for _ in range(i + 1, n)] for i in range(n)]

    mat_rot = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            angle_rad = angles[i][j - i - 1] / 180 * np.pi
            if flip:
                angle_rad = -angle_rad
            mat_plane_rot = np.eye(n)
            mat_plane_rot[i, i] = np.cos(angle_rad)
            mat_plane_rot[i, j] = -np.sin(angle_rad)
            mat_plane_rot[j, i] = np.sin(angle_rad)
            mat_plane_rot[j, j] = np.cos(angle_rad)

            mat_rot = mat_plane_rot @ mat_rot
    return mat_rot


def generate_n_state_flows(n, snr_param=2, n_entity=5, n_sample=5000, seed=None, ref_time=np.datetime64('2023-06-06'),
                           date_col=' Timestamp', label_col=' Label', src_feature_col=' Total Fwd Packets',
                           dst_feature_col=' Total Backward Packets', **kwargs):
    """
    Generates flow data with random n states. Each state has a particular entity relational behavior

    :param n: Number of network states
    :param snr_param: A parameter that controls signal to noise ratio
    :param n_entity: Number of entities in the network
    :param n_sample: Number of entity-wise samples
    :param seed: If True, set seed for all rng
    :param ref_time: Starting datetime of flows
    :param date_col: Flow data DataFrame column name indicating the starting datetime of the flow
    :param label_col: The dataframe flow_df column that holds the labels of flows
    :param src_feature_col: Flow data DataFrame column name indicating the source entity of the flow
    :param dst_feature_col: Flow data DataFrame column name indicating the destination entity of the flow
    :param kwargs: samples2flows kwargs
    :return: flow_df, f_mats
        flow_df: DataFrame containing flows
        f_mats: List of network state matrices
    """

    if seed:
        np.random.seed(seed)

    flow_df = pd.DataFrame()
    flow_t = copy.copy(ref_time)
    f_mats = []
    for i in range(n):
        samples_df, cov = generate_random_mvn_model(n_entity=n_entity, n_sample=n_sample)
        f_mat, _ = get_mat_f_q_from_covariance(cov)
        df_raw = samples2flows(samples_df, date_col=date_col, ref_time=flow_t, src_feature_col=src_feature_col,
                               dst_feature_col=dst_feature_col, **kwargs)
        df_raw[label_col] = str(i)
        temp_df = preprocess_df(df_raw, date_col=date_col)

        flow_df = pd.concat((flow_df, temp_df), ignore_index=True)
        f_mats.append(f_mat)
        flow_t = temp_df.iloc[-1][date_col]
    noise_std = 1 / snr_param
    noise = np.random.normal(scale=noise_std, size=2 * flow_df.shape[0])
    flow_df[src_feature_col] += noise[:flow_df.shape[0]]
    flow_df[dst_feature_col] += noise[-flow_df.shape[0]:]
    return flow_df, f_mats


def simulate_changing_network(p_other=0.4, rot_angle=40, n_entity=5, n_sample=200, contamination=0.01, n_blocks=10,
                              ref_time=np.datetime64('2023-06-06'), date_col=' Timestamp', label_col=' Label',
                              src_feature_col=' Total Fwd Packets', dst_feature_col=' Total Backward Packets',
                              **kwargs):
    """
    Generate background traffic according to unknown but fixed network state and randomly change the state for the
    next timeblock.

    :param p_other: Probability of having other network state for the current time block
    :param rot_angle: Angle that is used to rotate the covariance matrix of the network state
    :param n_entity: Number of entities in the network
    :param n_sample: Number of entity-wise samples
    :param contamination: Percentage of the flows that have the wrong label in a time block
    :param n_blocks: Total duration of blocks
    :param ref_time: Starting datetime of flows
    :param date_col: Flow data DataFrame column name indicating the starting datetime of the flow
    :param label_col: The dataframe flow_df column that holds the labels of flows
    :param src_feature_col: Flow data DataFrame column name indicating the source entity of the flow
    :param dst_feature_col: Flow data DataFrame column name indicating the destination entity of the flow
    :param kwargs: samples2flows kwargs
    :return: flow_df, network_states
        flow_df: DataFrame containing flows
        network_states: List of network state graph's adjacency matrices
    """
    _, background_state_cov = generate_random_mvn_model(n_entity=n_entity, n_sample=1)
    mat_rot = get_n_dim_rotation_matrix(n_entity, rot_angle)
    other_state_cov = mat_rot @ background_state_cov @ mat_rot.T

    flow_df = pd.DataFrame()
    flow_t = copy.copy(ref_time)
    network_states = []
    for i in range(n_blocks):
        p = np.random.random()
        current_state_cov = other_state_cov if p < p_other else background_state_cov
        f_current, _ = get_mat_f_q_from_covariance(current_state_cov)
        network_states.append(f_current)

        samples = nr.multivariate_normal(nr.rand(n_entity), current_state_cov, size=n_sample)
        samples_df = pd.DataFrame(samples, columns=list(range(n_entity)))
        df_raw = samples2flows(samples_df, date_col=date_col, ref_time=flow_t, src_feature_col=src_feature_col,
                               dst_feature_col=dst_feature_col, **kwargs)
        n_flow = df_raw.shape[0]

        attack_indicators = [1 if p < p_other else 0 for _ in range(n_flow)]
        # Contaminate labels
        attack_indicators = [str(1 - ind) if np.random.random() < contamination else str(ind) for ind in
                             attack_indicators]
        df_raw[label_col] = attack_indicators
        flow_t = df_raw[date_col].values[-1]

        temp_df = preprocess_df(df_raw, date_col=date_col)
        flow_df = pd.concat((flow_df, temp_df), ignore_index=True)

    return flow_df, network_states


if __name__ == '__main__':

    df_test, f_list = generate_n_state_flows(2, snr_param=1, max_flow_sep=0.4, min_flow_sep=0.2, min_flow_per_sample=5,
                                             max_flow_per_sample=10, n_sample=10_000, date_col='t', label_col='y')

    """
    df_test, f_list = simulate_changing_network(n_sample=10_000, min_flow_per_sample=5, max_flow_per_sample=10, date_col='t',
                                                label_col='y')
    """
    print(df_test)

    models = {'Linear Support Vector Machines': LinearSVC(dual='auto'), 'Decision Tree': DecisionTreeClassifier(),
              'Random Forest': RandomForestClassifier(), 'Naive Bayes': GaussianNB()}

    roc_curves = {}
    df_test_model = nre_classification(df_test, models, date_col='t', label_col='y', benign_label='0',
                                       t_graph=1000, time_scale='sec', standardize=False, sync_window_size=20,
                                       conn_param=None, roc_curves=roc_curves)
    fig = plot_roc_curves(roc_curves, title='Network Risk Estimation ROC Curve')

    #  TODO: Fix plotting for this test, low ROC points case
    print(df_test_model)
    plt.show()
