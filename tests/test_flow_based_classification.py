import numpy as np
import pandas as pd
import seaborn as sns
import copy
from matplotlib import pyplot as plt

from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from src.classification_tools import plot_roc_curves
from src.preprocess import preprocess_df
from src.analyze_cic_ids import flow_based_classification


def generate_n_flow_modes(n, n_flow=10000, flow_feat_mean_high=10, flow_feat_var=1, noise_std=5,
                          ref_time=np.datetime64('2023-06-06'), date_col=' Timestamp', label_col=' Label',
                          feature_col=' Total Fwd Packets', max_flow_sep=0.21, min_flow_sep=0.2):
    """
    Each mode has its own flow distribution

    :param n: Number of modes
    :param n_flow: Number of flows per mode
    :param flow_feat_mean_high: Maximum possible mean of flow feature.
        Feature mean is picked uniformly from [0, flow_feat_mean_high] for each flow.
    :param flow_feat_var: Variance of the flow feature.
        Flow feature drawn as a gaussian with mean given according to flow_feat_mean_high and with variance
        flow_feat_var.
    :param noise_std: Standard variation of the noise in flow feature
    :param ref_time: Starting datetime of flows
    :param date_col: Flow data DataFrame column name indicating the starting datetime of the flow
    :param label_col: The dataframe flow_df column that holds the labels of flows
    :param feature_col: The dataframe flow_df column that holds the flow features/attributes
    :param max_flow_sep: Maximum time interval between consecutive flows (in seconds)
    :param min_flow_sep: Minimum time interval between consecutive flows (in seconds)
    :return: flow_df, flow_means
        flow_df: DataFrame containing flows
        flow_means: List of flow mode means
    """
    flow_df = pd.DataFrame()
    flow_t = copy.copy(ref_time)
    flow_means = []
    for i in range(n):
        flow_feat_mean = np.random.uniform(high=flow_feat_mean_high)
        flow_feats = np.random.normal(flow_feat_mean, flow_feat_var, n_flow)
        flow_means.append(flow_feat_mean)

        dt = np.timedelta64(int(np.random.uniform(min_flow_sep, max_flow_sep) * 1000), 'ms')
        end_t = flow_t + dt * n_flow
        t = pd.date_range(flow_t, end_t, n_flow)
        flow_t = end_t

        df_raw = pd.DataFrame({date_col: t, feature_col: flow_feats, label_col: str(i)})
        temp_df = preprocess_df(df_raw, date_col=date_col)
        flow_df = pd.concat((flow_df, temp_df), ignore_index=True)
    noise = np.random.normal(scale=noise_std, size=n * n_flow)
    flow_df[feature_col] += noise

    return flow_df, flow_means


def simulate_attack(p_attack=0.4, n_flow=10000, contamination=0.1, n_blocks=100,
                    ref_time=np.datetime64('2023-06-06'), date_col=' Timestamp', label_col=' Label',
                    feature_col=' Total Fwd Packets', flow_feat_mean_high=20, flow_feat_var=1,
                    max_flow_sep=0.21, min_flow_sep=0.2):
    """
    Simulate a network traffic with background flows an occasionally attacks. Overall network flows are made up from
    consecutive 'time blocks'.

    :param p_attack: Probability of having an attack block
    :param n_flow: Number of flows per mode
    :param contamination: Percentage of the flows that have the wrong label in a time block
    :param n_blocks: Total duration of blocks
    :param ref_time: Starting datetime of flows
    :param date_col: Flow data DataFrame column name indicating the starting datetime of the flow
    :param label_col: The dataframe flow_df column that holds the labels of flows
    :param feature_col: The dataframe flow_df column that holds the flow features/attributes
    :param flow_feat_mean_high: Maximum possible mean of flow feature.
        Feature mean is picked uniformly from [0, flow_feat_mean_high] for each flow.
    :param flow_feat_var: Variance of the flow feature.
        Flow feature drawn as a gaussian with mean given according to flow_feat_mean_high and with variance
        flow_feat_var.
    :param max_flow_sep: Maximum time interval between consecutive flows (in seconds)
    :param min_flow_sep: Minimum time interval between consecutive flows (in seconds)

    :return: DataFrame containing flows
    """
    background_mean = np.random.uniform(high=flow_feat_mean_high)
    attack_mean = np.random.uniform(high=flow_feat_mean_high)

    flow_df = pd.DataFrame()
    flow_t = copy.copy(ref_time)
    flow_means = []
    for i in range(n_blocks):
        p = np.random.random()
        flow_feat_mean = attack_mean if p < p_attack else background_mean
        flow_feats = np.random.normal(flow_feat_mean, flow_feat_var, n_flow)
        flow_means.append(flow_feat_mean)
        attack_indicators = [1 if p < p_attack else 0 for _ in range(n_flow)]
        # Contaminate labels
        attack_indicators = [1 - ind if np.random.random() < contamination else ind for ind in attack_indicators]
        attack_indicators = [str(ind) for ind in attack_indicators]

        dt = np.timedelta64(int(np.random.uniform(min_flow_sep, max_flow_sep) * 1000), 'ms')
        end_t = flow_t + dt * n_flow
        t = pd.date_range(flow_t, end_t, n_flow)
        flow_t = end_t

        df_raw = pd.DataFrame({date_col: t, feature_col: flow_feats, label_col: attack_indicators})
        temp_df = preprocess_df(df_raw, date_col=date_col)
        flow_df = pd.concat((flow_df, temp_df), ignore_index=True)

    return flow_df, flow_means


if __name__ == '__main__':
    # Two different schemes to generate flows.
    #  df_test, means = simulate_attack(p_attack=0.5, n_flow=1000)
    df_test, _ = generate_n_flow_modes(2, flow_feat_mean_high=10, flow_feat_var=1, noise_std=5)

    print(df_test)
    sns.kdeplot(data=df_test, x=' Total Fwd Packets', hue=' Label', fill=True, alpha=0.5, linewidth=0)
    plt.title('Flow Features')

    models = {'Linear Support Vector Machines': LinearSVC(dual='auto'), 'Decision Tree': DecisionTreeClassifier(),
              'Random Forest': RandomForestClassifier(), 'Naive Bayes': GaussianNB()}

    roc_curves = {}
    df_test_model = flow_based_classification(df_test, models, benign_label='0', t_graph=10, roc_curves=roc_curves,
                                              feat_cols=[' Total Fwd Packets'])
    fig = plot_roc_curves(roc_curves, title='Flow-Based State Inference ROC Curve')

    print(df_test_model)
    plt.show()
