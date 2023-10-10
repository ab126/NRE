import pickle
import time

import pandas as pd
from matplotlib import pyplot as plt

from src.kalman_network_tools import plot_kalman_res
from src.preprocess import preprocess_df
from src.real_time_model import NetworkModelUnit
from src.time_windowed import get_window

file_addr = '..\CIC-IDS-2017\GeneratedLabelledFlows\TrafficLabelling\Monday-WorkingHours.pcap_ISCX.csv'
df_cic = pd.read_csv(file_addr, header=0, encoding='cp1252')
df = preprocess_df(df_cic, date_col=' Timestamp')

with open(r'saves\victim_net.pickle', 'rb') as handle:
    entity_names = pickle.load(handle)

unit = NetworkModelUnit(entity_names)
end_of_df = False
i = 0
date_col = ' Timestamp'
t_graph = 10  # 'sec'
t_update = 2

current_datetime = df.iloc[0][date_col]
last_datetime = df.iloc[-1][date_col]

date_times = [current_datetime]
fig = plt.figure()

while end_of_df is False:
    temp_df, current_datetime = get_window(current_datetime, df, date_col=date_col, time_window=t_graph,
                                           time_scale='sec')
    date_times.append(current_datetime)
    if current_datetime >= last_datetime:
        end_of_df = True
    unit.update_new_tick(temp_df)
    mat_x, mat_p = unit.mat_x, unit.mat_p

    plt.clf()
    plot_kalman_res(mat_x, mat_p, fig=fig)
    plt.draw()
    plt.pause(t_update)
