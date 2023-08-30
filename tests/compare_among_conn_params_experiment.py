import pandas as pd
import pickle

from matplotlib import pyplot as plt

from src.network_model import cic_conn_param_specs
from src.preprocess import preprocess_df
from src.analyze_cic_ids import compare_among_conn_params, plot_perf_comparison, nre_classification, \
    flow_based_classification

from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

file_addr = '..\CIC-IDS-2017\GeneratedLabelledFlows\TrafficLabelling\Tuesday-WorkingHours.pcap_ISCX.csv'  # 'Monday-WorkingHours.pcap_ISCX.csv' #  Wednesday-workingHours.pcap_ISCX.csv
df_cic = pd.read_csv(file_addr, header=0)

df = preprocess_df(df_cic, date_col=' Timestamp')
print(df.shape)

with open('saves/internal_nodes_tuesday.pickle', 'rb') as handle:
    entity_names = pickle.load(handle)

t_graph = 180  # s
labelling_opt = 'attacks first'
conn_param = 'Num Packets Rec'
feat_cols = (' Total Fwd Packets', ' Total Backward Packets')

all_df = compare_among_conn_params(df, entity_names, t_graph=t_graph, sync_window_size=1.2, best_op_point=True,
                                   )#conn_params=['Num Packets Rec'])
plot_perf_comparison(all_df, peak_only=False)
plt.show()
