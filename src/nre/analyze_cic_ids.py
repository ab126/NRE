import warnings

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, accuracy_score, precision_score, recall_score, balanced_accuracy_score, f1_score

from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from tqdm import tqdm

from .preprocess import preprocess_df
from .time_windowed import get_window
from .network_connectivity import MIN_SAMPLES, cic_conn_param_specs
from .kalman_network_tools import get_risk_mat_from_df
from .classification_tools import max_ba_operating_point, get_ba_from_operating_point, infer_roc

all_conn_params = ['NPS', 'NPR', 'Packet Length', 'Flow Duration', 'Flow Speed', 'Port Number',  # 'Activation',
                   'Protocol', 'Response Time', 'Packet Delay', 'Header Length', 'NAP', 'Active Time', 'Idle Time']
conn_param_name_dict = {'NPS': 'Number of Packets Sent', 'NPR': 'Number of Packets Received',
                        'NAP': 'Number of Active Packets'}  # Name dictionary


# all_conn_params = list(cic_conn_param_specs.keys())

def read_cic_flow_meter_csv(file_addr):
    """
    Reads the csv file that is CICFlowMeter description of network flows

    :param file_addr: Csv file address of the source data that is to be read

    :return:
        df : pandas.dataframe that contains CICFlowMeter features of each flow in source data
    """
    df_cic = pd.read_csv(file_addr, header=0)

    df = preprocess_df(df_cic, date_col=' Timestamp')
    return df


def split_df(df, ):
    """Splits the flow DataFrame df into consecutive parts given by split_ratio"""


def flow_data_parser(df, entity_names=None, sub_net_size=None, feat_cols=(' Total Fwd Packets',), date_col=' Timestamp',
                     t_graph=100, time_scale='sec', label_col=' Label', labelling_opt='attacks first',
                     benign_label='BENIGN', src_id_col=' Source IP', dst_id_col=' Destination IP', seed=None):
    """
    Parses the connection data into a data matrix using time windows.

    :param df: Canonical source dataframe that has flow information with timestamp, label and flow features. Each row is a flow
    :param entity_names: List of entities that the flows are constrained to
    :param sub_net_size: Percentage of the size of the total dataset is confined to. Overrides entity_names
    :param feat_cols: Column names of the source dataframe that is considered
    :param date_col: The dataframe df column that holds datetime timestamps of flows
    :param t_graph: Time window length that corresponds to a single state graph in time_scale units
    :param time_scale: Time window units. Either 'sec' or 'min'
    :param label_col: The dataframe df column that holds the labels of flows
    :param labelling_opt: Labelling scheme to be used; either 'attacks first' or 'majority'. 'attacks first' labels the
        windows that have a flow anything other than BENIGN as 'attack'. 'majority' option casts a majority vote
        among flows in the window to produce the label
    :param benign_label: Label of the benign, non-malicious, flows
    :param src_id_col: Column name of the flows data DatatFrame that identifies the source entity.
    :param dst_id_col: Column name of the flows data DatatFrame that identifies the destination entity.
    :param seed: If int, uses this seed for sub-sampling the entities if applicable

    :return: flow_data, labels, label_counts, flow_labels
        flow_data: The list of features of windowed flows in the same window
        labels: list[str]: List of Network State labels of the respective group of flows. Contains 'Empty' label if too few flows
            are present in the respective time window
        label_counts: List of flow counts of the respective group of flows
    """
    assert df.shape[0] > 0, "Source DataFrame is empty."

    if sub_net_size is not None:
        if seed:
            rand_state = seed
        else:
            rand_state = np.random.randint(0, 10 ** 5)
        np.random.seed(rand_state)

        # Stratify subsample
        df_benign = df[df[label_col] == benign_label]
        df_attack = df[df[label_col] != benign_label]

        all_benign_names = np.unique(np.concatenate((df_benign[src_id_col].values, df_benign[dst_id_col].values)))
        all_attack_names = np.unique(np.concatenate((df_attack[src_id_col].values, df_attack[dst_id_col].values)))
        size_benign = int(len(all_benign_names) * sub_net_size)
        size_attack = int(len(all_attack_names) * sub_net_size) * 2
        if size_attack > len(all_attack_names):
            size_attack = len(all_attack_names)

        benign_names = np.random.choice(all_benign_names, size=size_benign, replace=False)
        attack_names = np.random.choice(all_attack_names, size=size_attack, replace=False)
        entity_names = list(benign_names) + list(attack_names)

    if entity_names:
        idx = df[src_id_col].isin(entity_names) | df[dst_id_col].isin(entity_names)
        df = df[idx].copy()

    flow_data = []
    labels = []
    label_counts = []

    end_of_df = False
    current_datetime = df.iloc[0][date_col]
    last_datetime = df.iloc[-1][date_col]
    while end_of_df is False:
        temp_data = []
        temp_df, _, current_datetime = get_window(current_datetime, df, time_window=t_graph, date_col=date_col,
                                                  time_scale=time_scale, return_next_time=True)
        if current_datetime >= last_datetime:
            end_of_df = True
        if temp_df.empty or len(temp_df.shape) < 2 or temp_df.shape[0] < MIN_SAMPLES:
            flow_data.append(temp_data)
            labels.append('Empty')
            label_counts.append({})
            continue

        temp_data = [x for x in temp_df.loc[:, feat_cols].apply(pd.to_numeric).values]
        flow_data.append(temp_data)

        # Label Counting & Majority Labeling
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

    return flow_data, labels, label_counts


def _form_xy_ml(flow_data, labels, benign_label='BENIGN', test_size=None, seed=None):
    """
    Forms the data matrices and labels for Flow-based State Inference method Machine Learning models
    """
    # Form data matrices in accordance w/ windowing
    n_feats = len(flow_data[0][0])
    ind = np.array(labels) != 'Empty'
    x_win = [win_list for i, win_list in enumerate(flow_data) if ind[i]]  # non-empty flows grouped by windows
    y_win = np.array(labels)[ind]
    y_win = np.array([-1 if val == benign_label else 1 for val in y_win])

    if test_size is None:
        n_flow = [len(lst) for lst in x_win]  # Number of flow counts in each window
        x_flow = []  # individual flow data
        y_flow = []
        [x_flow.extend(lst) for lst in x_win]
        [y_flow.extend([lbl for _ in range(n_flow[j])]) for j, lbl in enumerate(y_win)]
        x_flow = np.array(x_flow).reshape(-1, n_feats)
        return x_flow, y_flow, n_flow, y_win
    else:
        ind = np.arange(len(x_win))
        rand_state = seed if seed else np.random.randint(0, 10 ** 5)
        ind_train, ind_test, y_win_train, y_win_test = train_test_split(ind, y_win, test_size=test_size,
                                                                        random_state=rand_state, stratify=y_win)
        x_win_train = [x_win[i] for i in ind_train]
        x_win_test = [x_win[i] for i in ind_test]

        n_flow_train = [len(lst) for lst in x_win_train]  # Number of flow counts in each window
        n_flow_test = [len(lst) for lst in x_win_test]
        x_flow_train = []  # individual flow data
        x_flow_test = []
        y_flow_train = []
        [x_flow_train.extend(lst) for lst in x_win_train]
        [x_flow_test.extend(lst) for lst in x_win_test]
        [y_flow_train.extend([lbl for _ in range(n_flow_train[j])]) for j, lbl in enumerate(y_win_train)]
        x_flow_train = np.array(x_flow_train).reshape(-1, n_feats)
        x_flow_test = np.array(x_flow_test).reshape(-1, n_feats)
        return x_flow_train, y_flow_train, n_flow_train, x_flow_test, y_win_test, n_flow_test


def flow_based_classification(df, models, entity_names=None, test_df=None,
                              feat_cols=(' Total Fwd Packets', ' Total Backward Packets'),
                              benign_label='BENIGN', labelling_opt='attacks first', standardize=False, seed=None,
                              test_size=0.33, roc_curves=None, warn=True, **kwargs):
    """
    Binary classification performance of flow-based classifier. Learns decision function on flows and
    then casts a majority vote on a window of flows for state inference (same windows as graph). For attack-first labeling
    option returned result dataframe might not be at a desirable operating point.

    :param df: Canonical source dataframe that has flow information with timestamp, label and flow features. Each row is a flow
    :param models: model dictionary (name:classifier) to use in state identification
    :param entity_names: List of entities that the flows are constrained to
    :param test_df: Test data in Canonical dataframe format
    :param feat_cols: Column names of the source dataframe that is considered
    :param benign_label: Label of the benign, non-malicious, flows
    :param labelling_opt: Labeling scheme to be used; either 'attacks first' or 'majority'. 'attacks first' labels the
        windows that have a flow anything other than BENIGN as 'attack'. 'majority' option casts a majority vote
        among flows in the window to produce the label
    :param standardize: If True, standardizes every flow feature among all training data flows
    :param seed: If int, uses this seed for train test split and for sub-sampling the entities if applicable
    :param test_size: Portion of flow data used as test
    :param roc_curves: (only if plot_roc==True and smooth_roc==True) If given dictionary, adds the model roc_curves
    :param warn: If True warn about limitations of printed results and redirect to roc_curves
    :param kwargs: flow_based_parser keyword arguments

    :return:
        df_model: DataFrame containing classification performance of all models using flow-based approach
    """

    if feat_cols is None:
        feat_cols = ('Active Mean', ' Flow Duration', 'Flow Bytes/s', ' Fwd Header Length', ' Bwd Header Length',
                     'Idle Mean', ' act_data_pkt_fwd', ' Total Fwd Packets', ' Total Backward Packets', ' Fwd IAT Mean',
                     ' Bwd IAT Mean', ' Fwd Packet Length Mean', ' Bwd Packet Length Mean', ' Source Port',
                     ' Destination Port', ' Protocol', ' Bwd IAT Mean', ' Fwd IAT Mean')

    flow_data, labels, label_counts = flow_data_parser(df, entity_names=entity_names, feat_cols=feat_cols,
                                                       benign_label=benign_label, labelling_opt=labelling_opt,
                                                       seed=seed, **kwargs)

    if test_df is not None:
        flow_data_test, labels_test, label_counts_test = flow_data_parser(test_df, entity_names=entity_names,
                                                                          feat_cols=feat_cols,
                                                                          benign_label=benign_label, seed=seed,
                                                                          labelling_opt=labelling_opt, **kwargs)

        x_flow_train, y_flow_train, n_flow_train, _ = _form_xy_ml(flow_data, labels,
                                                                  benign_label=benign_label,
                                                                  seed=seed)

        x_flow_test, _, n_flow_test, y_win_test = _form_xy_ml(flow_data_test, labels_test,
                                                              benign_label=benign_label,
                                                              seed=seed)
    else:
        x_flow_train, y_flow_train, n_flow_train, \
            x_flow_test, y_win_test, n_flow_test = _form_xy_ml(flow_data, labels,
                                                               benign_label=benign_label,
                                                               test_size=test_size,
                                                               seed=seed)

    if standardize:
        ss_train = StandardScaler()
        x_flow_train = ss_train.fit_transform(x_flow_train)
        x_flow_test = ss_train.transform(x_flow_test)

    # Performance
    accuracy, precision, recall, b_acc, f1 = {}, {}, {}, {}, {}
    for mdl in models.keys():
        # Fit the classifier
        models[mdl].fit(x_flow_train, y_flow_train)

        # Make predictions & custom ROC
        y_flow_predict = models[mdl].predict(x_flow_test)
        y_win_predict = []
        scores = []
        ind = 0

        # Decision Aggregation
        for n in n_flow_test:
            y_temp = y_flow_predict[ind: ind + n]
            vals, counts = np.unique(y_temp, return_counts=True)

            ind += n
            pos_prob = counts[vals == 1][0] / n if len(counts[vals == 1]) > 0 else 0
            scores.append(pos_prob)

            if labelling_opt == 'score':
                ind2 = np.argwhere(vals == 1).flatten()
                p = counts[ind2][0] / len(y_temp) if len(counts[ind2]) != 0 else 0
                lbl = 1 if np.random.rand() < p else -1
                y_win_predict.append(lbl)
            elif labelling_opt == 'attacks first':
                if 1 in vals:
                    y_win_predict.append(1)
                else:
                    y_win_predict.append(-1)
            else:  # Majority labelling
                i = np.argmax(counts)  # Majority Voting
                y_win_predict.append(vals[i])

        if roc_curves is not None:
            fpr, tpr, thresholds = roc_curve(y_win_test, scores)
            assert type(roc_curves) == dict, "roc_curves must be a dictionary"
            roc_curves[mdl] = fpr, tpr

        # Calculate metrics
        accuracy[mdl] = accuracy_score(y_win_test, y_win_predict)
        precision[mdl] = precision_score(y_win_test, y_win_predict)
        recall[mdl] = recall_score(y_win_test, y_win_predict)
        b_acc[mdl] = balanced_accuracy_score(y_win_test, y_win_predict)
        f1[mdl] = f1_score(y_win_test, y_win_predict)

    df_model = pd.DataFrame(index=models.keys(), columns=['Accuracy', 'Precision', 'Recall'])
    df_model['Accuracy'] = accuracy.values()
    df_model['Precision'] = precision.values()
    df_model['Recall'] = recall.values()
    df_model['Balanced Accuracy'] = b_acc.values()
    df_model['f1'] = f1.values()

    if warn:
        warnings.warn('Reported results are not fully representative since thresholding is not implemented for the ' +
                      'flow_based_classification function directly. Use the roc_curves parameters to gauge the' +
                      ' performance instead.')

    return df_model


def nre_classification(df, models, test_df=None, standardize=False, benign_label='BENIGN', roc_curves=None,
                       test_size=0.33, seed=None, **kwargs):
    """
    Binary classification performance of Network Risk Estimation Method.

    :param df: Canonical source dataframe that has flow information with timestamp, label and flow features. Each row
        is a flow
    :param models: model dictionary (name:classifier) to use in state identification
    :param test_df: Test data in Canonical dataframe format
    :param standardize: If True, Standardizes risk estimate data in classification:param benign_label: Label of the benign, non-malicious, flows
    :param benign_label: Label of the benign, non-malicious, flows
    :param roc_curves: (only if plot_roc==True and smooth_roc==True) If given dictionary, adds the model roc_curves
    :param test_size: Portion of flow data used as test
    :param seed: If int, uses this seed for train test split
    :param kwargs: get_risk_mat_from_df keyword arguments
    :return:
        df_model: DataFrame containing classification performance of all models using NRE approach

   """
    risk_mat, labels, _, _, _ = get_risk_mat_from_df(df, benign_label=benign_label, **kwargs)
    ind = np.array(labels) != 'Empty'

    if test_df is not None:
        risk_mat_test, labels_test, _, _, _ = get_risk_mat_from_df(test_df, benign_label=benign_label,
                                                                   **kwargs)
        ind_test = np.array(labels_test) != 'Empty'

        mat_x_train = risk_mat[ind, :]
        y_train = np.array([-1 if val == benign_label else 1 for val in np.array(labels)[ind]])
        mat_x_test = risk_mat_test[ind_test, :]
        y_test = np.array([-1 if val == benign_label else 1 for val in np.array(labels_test)[ind_test]])
    else:
        mat_x = risk_mat[ind, :]
        y = np.array(labels)[ind]
        y_bin = np.array([-1 if val == benign_label else 1 for val in y])
        rand_state = seed if seed else np.random.randint(0, 10 ** 5)
        mat_x_train, mat_x_test, y_train, y_test = train_test_split(mat_x, y_bin, test_size=test_size,
                                                                    random_state=rand_state, stratify=y_bin)

    if standardize:
        ss_train = StandardScaler()
        mat_x_train = ss_train.fit_transform(mat_x_train)

        # ss_test = StandardScaler()
        mat_x_test = ss_train.transform(mat_x_test)

    accuracy, precision, recall, b_acc, f1, auc_score = {}, {}, {}, {}, {}, {}
    for mdl in models.keys():
        # Fit the classifier
        models[mdl].fit(mat_x_train, y_train)
        # Make predictions
        y_predict = models[mdl].predict(mat_x_test)
        # Calculate metrics
        accuracy[mdl] = accuracy_score(y_test, y_predict)
        precision[mdl] = precision_score(y_test, y_predict)
        recall[mdl] = recall_score(y_test, y_predict)
        b_acc[mdl] = balanced_accuracy_score(y_test, y_predict)
        f1[mdl] = f1_score(y_test, y_predict)

        if roc_curves is not None:
            if mdl == 'Linear Support Vector Machines':
                scores = models[mdl].decision_function(mat_x_test)
            else:
                scores = models[mdl].predict_proba(mat_x_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, scores)
            assert type(roc_curves) == dict, "roc_curves must be a dictionary"
            roc_curves[mdl] = fpr, tpr

    df_model = pd.DataFrame(index=models.keys(), columns=['Accuracy', 'Precision', 'Recall'])
    df_model['Accuracy'] = accuracy.values()
    df_model['Precision'] = precision.values()
    df_model['Recall'] = recall.values()
    df_model['Balanced Accuracy'] = b_acc.values()
    df_model['f1'] = f1.values()

    return df_model


# TODO: Both parsers might not be synced
def compare_among_conn_params(df, models=None, entity_names_nre=None, entity_names_fb=None, sub_net_size=None,
                              conn_params=None, t_graph=100, sync_window_size=1, time_scale='sec', standardize=False,
                              best_op_point=True,
                              seed=None, control=False, **kwargs):
    """
    Compares NRE method with common flow-based classifier among different connection parameters

    :param df: Canonical source dataframe that has flow information with timestamp, label and flow features. Each row
        is a flow
    :param models: Machine Learning models to be used as classifiers in comparison
    :param entity_names_nre: List of entities that the flows are constrained to for NRE method
    :param entity_names_fb: List of entities that the flows are constrained to for Flow-based method. If None,
        entity_names_nre is used instead
    :param sub_net_size: Percentage of the size of the total dataset is confined to for fb method. Overrides entity_names
    :param conn_params: Predefined connection parameter that the samples are calculated for.
    :param t_graph: Time window length that corresponds to a single state graph in time_scale units
    :param sync_window_size: Time window size for synchronizing flows.
    :param time_scale: Time window units. Either 'sec' or 'min'
    :param standardize: If True, Standardizes risk estimate and flow features data
    :param best_op_point: If True, picks the operating point with the highest balanced accuracy and only audits balanced
        accuracy metric
    :param seed: If int, uses this seed for train test split and for sub-sampling the entities if applicable
    :param control: If True add the control for FBNSI method without confining entities
    :param kwargs: nre_classification kwargs
    :return all_df: Classification results as a DataFrame object
    """
    if models is None:
        models = {'Decision Trees': DecisionTreeClassifier(),
                  # 'Linear Support Vector Machines': LinearSVC(dual='auto')
                  'Random Forest': RandomForestClassifier(), 'Naive Bayes': GaussianNB()}
    if conn_params is None:
        conn_params = copy.copy(all_conn_params)

    all_df = pd.DataFrame()
    if control:
        # Create Feature Columns for all connection parameters
        feat_cols_all = [cic_conn_param_specs[conn_param]['src_feature_col'] for conn_param in all_conn_params]
        [feat_cols_all.append(cic_conn_param_specs[conn_param]['dst_feature_col']) for conn_param in all_conn_params]
        feat_cols_all = list(np.unique(feat_cols_all))
        print('All features: ', feat_cols_all)

        flow_based_curves_control = {}
        df_flow_control = flow_based_classification(df, models, entity_names=None,
                                                    feat_cols=feat_cols_all, t_graph=t_graph,
                                                    roc_curves=flow_based_curves_control,
                                                    time_scale=time_scale, standardize=standardize, seed=seed,
                                                    warn=not best_op_point)
        if best_op_point:
            df_flow_control = df_flow_control.loc[:, ['Balanced Accuracy']]
            for key in flow_based_curves_control:
                fpr, tpr = flow_based_curves_control[key]
                x_roc, y_roc = infer_roc(fpr, tpr)
                x_op, y_op, conv_bool = max_ba_operating_point(x_roc, y_roc)
                ba = get_ba_from_operating_point(x_op, y_op) if conv_bool else np.nan
                df_flow_control.loc[key, 'Balanced Accuracy'] = ba
        df_flow_control['Classifier'] = df_flow_control.index
        df_flow_control.index = np.arange(df_flow_control.shape[0])
        df_flow_control['Method'] = ['FBNSI Control' for _ in
                                     range(df_flow_control.shape[0])]

    for conn_param in tqdm(conn_params):
        nre_curves = {}
        df_nre = nre_classification(df, models, conn_param=conn_param, entity_names=entity_names_nre, t_graph=t_graph,
                                    verbose=False, sync_window_size=sync_window_size, roc_curves=nre_curves,
                                    time_scale=time_scale, standardize=standardize, seed=seed, **kwargs)
        if best_op_point:
            df_nre = df_nre.loc[:, ['Balanced Accuracy']]
            for key in nre_curves:
                fpr, tpr = nre_curves[key]
                x_roc, y_roc = infer_roc(fpr, tpr)
                x_op, y_op, conv_bool = max_ba_operating_point(x_roc, y_roc)
                ba = get_ba_from_operating_point(x_op, y_op) if conv_bool else np.nan
                df_nre.loc[key, 'Balanced Accuracy'] = ba

        conn_param_str = conn_param if conn_param not in conn_param_name_dict else conn_param_name_dict[conn_param]
        df_nre['Connection Parameter'] = [conn_param_str for _ in range(df_nre.shape[0])]
        df_nre['Classifier'] = df_nre.index
        df_nre.index = np.arange(df_nre.shape[0])
        df_nre['Method'] = ['NRE' for _ in range(df_nre.shape[0])]

        feat_cols = list(
            {cic_conn_param_specs[conn_param]['src_feature_col'], cic_conn_param_specs[conn_param]['dst_feature_col']})

        flow_based_curves = {}
        df_flow = flow_based_classification(df, models, entity_names=entity_names_fb,
                                            feat_cols=feat_cols, t_graph=t_graph, roc_curves=flow_based_curves,
                                            time_scale=time_scale, standardize=standardize, seed=seed,
                                            warn=not best_op_point, sub_net_size=sub_net_size)
        if best_op_point:
            df_flow = df_flow.loc[:, ['Balanced Accuracy']]
            for key in flow_based_curves:
                fpr, tpr = flow_based_curves[key]
                x_roc, y_roc = infer_roc(fpr, tpr)
                x_op, y_op, conv_bool = max_ba_operating_point(x_roc, y_roc)
                ba = get_ba_from_operating_point(x_op, y_op) if conv_bool else np.nan
                df_flow.loc[key, 'Balanced Accuracy'] = ba
        df_flow['Connection Parameter'] = [conn_param_str for _ in range(df_flow.shape[0])]
        df_flow['Classifier'] = df_flow.index
        df_flow.index = np.arange(df_flow.shape[0])
        df_flow['Method'] = ['FBNSI' for _ in range(df_flow.shape[0])]

        all_df = pd.concat((all_df, df_flow, df_nre), ignore_index=True)

        if control:
            dt_temp = df_flow_control.copy()
            dt_temp['Connection Parameter'] = [conn_param_str for _ in range(df_flow_control.shape[0])]
            all_df = pd.concat((all_df, dt_temp), ignore_index=True)

    return all_df, flow_based_curves


def plot_perf_comparison(all_df, title='', perf_metric='Balanced Accuracy', peak_only=False, **kwargs):
    """
    Plots the performance comparison between NRE and flow based classifiers
    """
    if peak_only:
        new_cols = ['Connection Parameter', 'Method', perf_metric]
        peak_df = pd.DataFrame(columns=new_cols)
        for method in np.unique(all_df['Method']):
            for conn_param in np.unique(all_df['Connection Parameter']):
                ind = (all_df['Method'] == method) & (all_df['Connection Parameter'] == conn_param)
                idx = all_df[ind][perf_metric].idxmax()
                temp_df = all_df.loc[idx, new_cols].to_frame().T
                peak_df = pd.concat((peak_df, temp_df), ignore_index=True)
        all_df = peak_df.rename(columns={perf_metric: 'Peak ' + perf_metric})
        perf_metric = 'Peak ' + perf_metric

    sns.barplot(data=all_df, x='Connection Parameter', y=perf_metric, hue='Method', **kwargs)
    plt.xticks(rotation=45, ha='right')
    plt.title(title)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    return plt.gcf()
