# Network Risk Estimation (NRE)
This code is provided as an implementation of our work 
network risk estimation (NRE) which is a real-time probabilistic 
risk estimator based on entity relationships in connection data 
and active measurements on a set of entities.

A Python API is offered for real-time operation.

## Python

### Installation
NRE has been tested with Python 3.10 and it requires:
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
also requires [npeet](https://github.com/gregversteeg/NPEET) [2]
package whose modified version is placed under [src.nre](https://github.com/ab126/NRE/tree/main/src/nre).


### Working Example
This example can be found in [test_working_example.py](https://github.com/ab126/NRE/blob/main/tests/test_working_example.py).

First, the connection data is read from a source
such as ".txt" file, and it is transformed to the desired format. For convenience, [CICFlowMeter](https://www.unb.ca/cic/research/applications.html#CICFlowMeter) [1]
format has been opted for ".txt" template for the connection data.

```python
import pandas as pd
import numpy as np
import time

from matplotlib import pyplot as plt

from src.nre import plot_kalman_res
from src.nre import preprocess_df
from src.nre import NetworkModel

df_raw = pd.read_csv('..\\test_flows.csv', header=0, encoding='cp1252')
df = preprocess_df(df_raw, date_col=' Timestamp')
```
Next, NetworkModel instance is initialized with initial risk estimates.

```python
nm = NetworkModel(entity_names=list(np.arange(5)), mat_x_init= np.ones(5), mat_p_init=np.eye(5))
```
Then, the canonical connection data is fed through the model to estimate the risks at the new time tick. See [ConnectivityUnit.read_flows](https://github.com/ab126/NRE/blob/main/src/network_connectivity.py)
for all the hyperparameters.

```python
nm.update_new_tick_conn_data(df, src_id_col=' Source ID', dst_id_col=' Destination ID',
                             src_feature_col=' Source Flow Attribute', dst_feature_col=' Destination Flow Attribute',
                             sync_window_size=20, time_scale='sec', keep_unit=True)
mat_x, mat_p = nm.mat_x, nm.mat_p
```

# References
1. Arash Habibi Lashkari, Gerard Draper-Gil, Mohammad Saiful Islam Mamun and Ali A. Ghorbani, "Characterization of Tor Traffic Using Time Based Features", In the proceeding of the 3rd International Conference on Information System Security and Privacy, SCITEPRESS, Porto, Portugal, 2017
2. Gregversteeg/NPEET: Non-parametric entropy estimation toolbox, GitHub. Available at: https://github.com/gregversteeg/NPEET (Accessed: 10 October 2024). 
