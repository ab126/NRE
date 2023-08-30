import pandas as pd
import pickle

from matplotlib import pyplot as plt

from src.preprocess import preprocess_df
from src.analyze_cic_ids import nre_classification, flow_based_classification
from src.classification_tools import max_ba_operating_point, get_ba_from_operating_point, plot_roc_curves

from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

file_addr = '..\CIC-IDS-2017\GeneratedLabelledFlows\TrafficLabelling\Tuesday-WorkingHours.pcap_ISCX.csv'  # 'Monday-WorkingHours.pcap_ISCX.csv' #  Wednesday-workingHours.pcap_ISCX.csv
df_cic = pd.read_csv(file_addr, header=0)

df = preprocess_df(df_cic, date_col=' Timestamp')
print(df.shape)

with open('saves/victim_net.pickle', 'rb') as handle:
    entity_names = pickle.load(handle)

with open('saves/partitioned_nodes_106.pickle', 'rb') as handle:
    subnet_names = pickle.load(handle)

models = {'Linear Support Vector Machines': LinearSVC(dual='auto'), 'Decision Tree': DecisionTreeClassifier(),
          'Random Forest': RandomForestClassifier(), 'Naive Bayes': GaussianNB()}
t_graph = 90  # s
labelling_opt = 'attacks first'
conn_param = 'Num Packets Received'
feat_cols = (' Total Fwd Packets', ' Total Backward Packets')

nre_curves = {}
df_nre = nre_classification(df, models, entity_names=subnet_names, t_graph=t_graph,
                            conn_param=conn_param, sync_window_size=1.2, roc_curves=nre_curves)
fig1 = plot_roc_curves(nre_curves, title='Network Risk Estimation ROC Curve')
print(df_nre)

flow_based_curves = {}
df_flow_based = flow_based_classification(df, models, entity_names=entity_names, t_graph=t_graph,
                                          roc_curves=flow_based_curves)
fig2 = plot_roc_curves(flow_based_curves, title='Flow-Based State Inference ROC Curve')
print(df_flow_based)
plt.show()



