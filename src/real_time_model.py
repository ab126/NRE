from src.preprocess import preprocess_df
from src.network_model import NetworkModel
from src.kalman_network_tools import single_step_update


# TODO: Write a class that implements all the necessary functions for a single graph window in another module

class NetworkModelUnit:
    """
        Basic Unit for updating risk estimates from previous estimates and connection data
    """

    def __init__(self, entity_names=None, sync_window_size=1.2, time_scale='sec', mat_x_init=None, mat_p_init=None,
                 mat_q=None):
        self.entity_names = entity_names
        self.sync_window_size = sync_window_size
        self.time_scale = time_scale
        self.mat_x = mat_x_init
        self.mat_p = mat_p_init
        self.mat_q = mat_q

    # Prototypical update method
    def update_new_tick(self, df_conn, measurement=None, mat_h=None, mat_r=None, **kwargs):
        """Update the risk estimates according to previous estimate"""

        df = preprocess_df(df_conn, date_col=' Timestamp')
        nm = NetworkModel()
        nm.read_flows(df, entity_names=self.entity_names, window_type='time', sync_window_size=self.sync_window_size,
                      time_scale=self.time_scale, **kwargs)
        nm.fit_graph_model(method='cov', verbose=True)

        self.mat_x, self.mat_p = single_step_update(nm.F, measurement=measurement, mat_h=mat_h, mat_x_init=self.mat_x,
                                                    mat_p_init=self.mat_p, mat_q=self.mat_q, mat_r=mat_r, k_steps=1,
                                                    relief_factor=0.6, normalize=False)



