import pandas as pd
import numpy as np
import time

from matplotlib import pyplot as plt

from src.nre.kalman_network_tools import plot_kalman_res
from src.nre.preprocess import preprocess_df
from src.nre.real_time_model import NetworkModel

start = time.time()
df_raw = pd.read_csv('test_flows.csv', header=0, encoding='cp1252')
df = preprocess_df(df_raw, date_col=' Timestamp')

end = time.time()
print('File to df time (s): ', end - start, '\n')

start = time.time()
nm = NetworkModel(entity_names=list(np.arange(5)), mat_x_init= np.ones(5), mat_p_init=np.eye(5))
nm.update_new_tick_conn_data(df, src_id_col=' Source ID', dst_id_col=' Destination ID',
                             src_feature_col=' Source Flow Attribute', dst_feature_col=' Destination Flow Attribute',
                             sync_window_size=20, time_scale='sec', keep_unit=True)
end = time.time()
print('\nDf to risk estimates (s): ', end - start)
mat_x, mat_p = nm.mat_x, nm.mat_p

print('\nSamples to Graph time (s): ', end - start)
print('\nNumber of Samples in Fitting:', nm.cu.samples.shape[0])
fig = plot_kalman_res(mat_x, mat_p, str_k='1')
plt.show()
