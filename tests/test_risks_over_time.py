import pickle

import numpy as np
import pandas as pd

from src.kalman_network_tools import graphs_2_risk_scores
from src.network_connectivity import ConnectivityUnit
from src.network_plotting import risks_over_time_3d
from src.preprocess import preprocess_df

file_addr = '..\CIC-IDS-2017\GeneratedLabelledFlows\TrafficLabelling\Monday-WorkingHours.pcap_ISCX.csv'
df_cic = pd.read_csv(file_addr, header=0, encoding='cp1252')

df = preprocess_df(df_cic, date_col=' Timestamp')
print("Shape of dataframe: ", df.shape)
df = df.iloc[:303_500, :].copy()

with open(r'saves\victim_net.pickle', 'rb') as handle:
    entity_names = pickle.load(handle)
print("Size of sub network: ", len(entity_names))

nm = ConnectivityUnit()
nm.read_flows(df, conn_param='Num Packets Received', entity_names=entity_names,
              window_type='time', sync_window_size=1.2, time_scale='sec')

nm.fit_connectivity_model(method='cov')  # cov
nm.plot_f(labels=True)
mthd_name = 'Corr. Coeff.'

nn = 4
all_graphs = [nm.mat_f for i in range(nn)]
all_graphs = np.array(all_graphs)
all_measurements = [None for _ in range(nn)]
all_measurements[1] = [5.6]
all_mat_h = [None for _ in range(nn)]
mat_h = np.zeros((1, len(entity_names)))
mat_h[0, 1] = 1
all_mat_h[1] = mat_h

x_list, P_list = graphs_2_risk_scores(all_graphs, all_measurements=all_measurements, all_mat_h=all_mat_h,
                                      k_steps=1, relief_factor=0.09, sequential=True, normalize=False, return_cov=True)
fig = risks_over_time_3d(x_list, P_list, t_graph=90, title='Victim Network Risk Estimates',
                         save_name='saves/risk_ests_3d_v3.jpg')


print("Entity names: ", entity_names)
