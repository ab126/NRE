# Network Risk Estimation (NRE)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21520902.svg)](https://doi.org/10.5281/zenodo.21520902)

This code is provided as an implementation of our work 
network risk estimation (NRE) which is a real-time probabilistic 
risk estimator based on entity relationships in connection data 
and active measurements on a set of entities.

A Python API is offered for real-time operation.

## Python

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
13. pingouin

For beyond correlative entity relationship modelling, NRE
also requires [npeet](https://github.com/gregversteeg/NPEET) [2]
package whose adopted version can be found at [nre/npeet](https://github.com/ab126/NRE/tree/main/nre/npeet).


## Installation

NRE requires Python 3.10 or later.

### Install directly from GitHub

```bash
python -m pip install git+https://github.com/ab126/NRE.git
```

### Install from a local clone

```bash
git clone https://github.com/ab126/NRE.git
cd NRE
python -m pip install .
```

### Verify the installation

```bash
python -c "import nre; print('NRE installed successfully')"
```

### Working Example
This example can be found in [test_working_example.py](https://github.com/ab126/NRE/blob/main/nre/tests/test_working_example.py).

First, the connection data is read from a source
such as ".txt" file, and it is transformed to the desired format. For convenience, [CICFlowMeter](https://www.unb.ca/cic/research/applications.html#CICFlowMeter) [1]
format has been opted for ".txt" template for the connection data.

```python
import pandas as pd
from nre.preprocess import preprocess_df

df_raw = pd.read_csv('..\\..\\test_flows.csv', header=0, encoding='cp1252')
df = preprocess_df(df_raw, date_col=' Timestamp')
```
Next, NetworkModel instance is initialized with initial risk estimates.

```python
from nre.real_time_model import NetworkModel

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

## Citation

If you use NRE in your research, please cite the software:

```bibtex
@software{bayer2026nre,
  author  = {Arda Bayer},
  title   = {Network Risk Estimation (NRE)},
  year    = {2026},
  version = {1.0.0},
  doi     = {10.5281/zenodo.21520902},
  url     = {https://doi.org/10.5281/zenodo.21520902}
}
```

## License

[MIT](https://choosealicense.com/licenses/mit/)

# References
1. Arash Habibi Lashkari, Gerard Draper-Gil, Mohammad Saiful Islam Mamun and Ali A. Ghorbani, "Characterization of Tor Traffic Using Time Based Features", In the proceeding of the 3rd International Conference on Information System Security and Privacy, SCITEPRESS, Porto, Portugal, 2017
2. Gregversteeg/NPEET: Non-parametric entropy estimation toolbox, GitHub. Available at: https://github.com/gregversteeg/NPEET (Accessed: 10 October 2024). 
