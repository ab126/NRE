import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.preprocess import preprocess_df
from src.network_connectivity import ConnectivityUnit
from src.analyze_cic_ids import nre_classification, flow_based_classification
from src.classification_tools import plot_roc_curves

from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from src.validation_tools import validate_model

file_addr = '..\CIC-IDS-2017\GeneratedLabelledFlows\TrafficLabelling\Tuesday-WorkingHours.pcap_ISCX.csv'
df_cic = pd.read_csv(file_addr, header=0, encoding='cp1252')

df = preprocess_df(df_cic, date_col=' Timestamp')

with open('saves/victim_net.pickle', 'rb') as handle:
    entity_names = pickle.load(handle)

val_ratio, test_ratio = 0.25, 0.25
assert val_ratio + test_ratio < 1, "No data left for training"


def model_nre(df_train, ml_models, **kwargs):
    return nre_classification(df_train, ml_models, entity_names=entity_names, verbose=True, **kwargs)


def model_fb(df_train, ml_models, **kwargs):
    return flow_based_classification(df_train, ml_models, entity_names=entity_names, **kwargs)


val_ratio, test_ratio = 0.25, 0.25
n_all = df_cic.shape[0]
n_train = int(n_all * (1 - val_ratio - test_ratio))
n_val = int(n_all * val_ratio)

df_train = df.iloc[:n_train, :]
df_val = df.iloc[n_train:n_train + n_val, :]
df_test = df.iloc[n_train + n_val:, :]

param_list = [{'t_graph': 90, 'sync_window_size': 1.2, 'forget_factor': 0.5, 'relief_factor': 0.6, 'k_steps': 1}]
auc_scores_nre = validate_model(df_train, df_val, model_nre, param_list)
