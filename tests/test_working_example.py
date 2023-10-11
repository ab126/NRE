import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

from src.network_model import NetworkModel
from src.preprocess import preprocess_df

start = time.time()
df_raw = pd.read_csv('..\\test_flows.csv', header=0, encoding='cp1252')
df = preprocess_df(df_raw, date_col=' Timestamp')

end = time.time()
print('File to df time (s): ', end - start)

start = time.time()
nm = NetworkModel()
nm.read_flows(df, entity_names=list(np.arange(5)), src_id_col=' Source ID', dst_id_col=' Destination ID',
              src_feature_col=' Source Flow Attribute', dst_feature_col=' Destination Flow Attribute',
              sync_window_size=20, time_scale='sec')
end = time.time()
print('\nDf to samples time (s): ', end - start, '\n')

start = time.time()
nm.fit_graph_model()
nm.plot_f()
end = time.time()
print('\nSamples to Graph time (s): ', end - start)
plt.title('Fitted Matrix F')
print('\nNumber of Samples in Fitting:', nm.samples.shape[0])