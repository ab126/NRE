from src.preprocess import preprocess_df
from src.network_connectivity import ConnectivityUnit
from src.kalman_network_tools import single_step_update


# TODO: Write a class that implements all the necessary functions for a single graph window in another module

class NetworkModel:
    """
        Base unit for updating risk estimates from previous estimates and connection data
    """

    def __init__(self, entity_names=None, mat_x_init=None, mat_p_init=None,
                 mat_q=None):
        """
        :param entity_names: List of entity names
        :param mat_x_init: Initial risk estimate as 1d np.array
        :param mat_p_init: Initial risk covariance matrix as 2d np.ndarray
        :param mat_q: System noise covariance matrix as 2d np.ndarray
        """
        self.cu = None
        self.entity_names = entity_names
        self.mat_x = mat_x_init
        self.mat_p = mat_p_init
        self.mat_q = mat_q

    def update_new_tick(self, df_conn, measurement=None, mat_h=None, mat_r=None, keep_unit=False, relief_factor=0.6,
                        **kwargs):
        """
        Update the risk estimates according to previous estimate

        :param df_conn: Canonical connection data DataFrame
        :param measurement: Risk measurements array
        :param mat_h: Observation matrix
        :param mat_r: Measurement noise covariance matrix
        :param keep_unit: If true stores the latest ConnectivityUnit as self.cu
        :param relief_factor: Percentage of the risk relieved at each time step for each node (see the paper for more)
        :param kwargs: ConnectivityUnit.read_flows kwargs
        :return: None
        """

        df = preprocess_df(df_conn, date_col=' Timestamp')
        cu = ConnectivityUnit()
        cu.read_flows(df, entity_names=self.entity_names, window_type='time', **kwargs)
        cu.fit_graph_model(method='cov', verbose=True)
        if keep_unit:
            self.cu = cu

        self.mat_x, self.mat_p = single_step_update(cu.F, measurement=measurement, mat_h=mat_h, mat_x_init=self.mat_x,
                                                    mat_p_init=self.mat_p, mat_q=self.mat_q, mat_r=mat_r, k_steps=1,
                                                    relief_factor=relief_factor, normalize=False)
