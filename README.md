# Network Risk Estimation (NRE)
This code is provided as an implementation of our work 
network risk estimation (NRE) which is a real-time probabilistic 
risk estimator based on entity relationships in connection data 
and active measurements on a set of entities.

A Python API is offered for real-time operation.

## Python

### Installation
NRE has been tested with Python 3.10 and requires:
1. numpy
2. pandas
3. scikit-learn
4. scipy
5. datetime
6. ordered-set
7. filterpy
8. tqdm
9. matplotlib
10. seaborn
11. networkx
12. pillow

These dependencies can be installed via
`
pip install -r requirements.txt
`.

For beyond correlative entity relationship modelling, it
also requires [npeet](https://github.com/gregversteeg/NPEET) 
package which can be placed under [src](https://github.com/ab126/NRE/tree/main/src).


### Working Example
This example can be found in [test_working_example.py](https://github.com/ab126/NRE/blob/main/tests/test_working_example.py)
```python
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
```

