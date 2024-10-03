import numpy as np

from .preprocess import preprocess_df
from .network_connectivity import ConnectivityUnit, single_risk_update


# TODO: This class should deal with the whole network. Add detailed docs (?)

class NetworkModel:
    """
        Base unit for updating risk estimates from previous estimates and connection data
    """

    def __init__(self, entity_names=None, mat_x_init=None, mat_p_init=None, mat_q=None):
        """
        :param entity_names: List of entity names
        :param mat_x_init: Initial risk estimate as 1d np.array
        :param mat_p_init: Initial risk covariance matrix as 2d np.ndarray
        :param mat_q: System noise covariance matrix as 2d np.ndarray
        """
        if entity_names is None:
            entity_names = []

        self.entity_names = entity_names
        self.cu = ConnectivityUnit() # Only for debugging for now
        self.mat_f = np.eye(len(entity_names))
        self.mat_x = np.zeros(0) if mat_x_init is None else mat_x_init
        self.mat_p = np.eye(0) if mat_p_init is None else mat_p_init
        self.mat_q = mat_q

    def normalize_risks(self):
        """ Map the risks to [0, 1] range """
        scale = self.mat_x.max() / 2
        self.mat_x = self.mat_x / scale
        self.mat_p = self.mat_p / scale ** 2

    def add_entities(self, entity_list):
        """ Adds from the list of entities. Added entities will have the mean of previous entity risk and uncorrelated
         covariance with mean old variance"""
        old_n = len(self.entity_names)
        new_entities = [name for name in entity_list if name not in self.entity_names]
        new_risk = np.mean(self.mat_x) if old_n != 0 else 1
        new_var = np.mean(np.diag(self.mat_p)) if old_n != 0 else 1

        self.entity_names = self.entity_names + new_entities
        self.cu = ConnectivityUnit()

        temp_f = np.eye(len(self.entity_names))
        temp_f[:old_n, :old_n] = self.mat_f
        self.mat_f = temp_f.copy()

        temp_x = np.ones(len(self.entity_names)) * new_risk
        temp_x[:old_n] = self.mat_x
        self.mat_x = temp_x.copy()

        temp_p = np.eye(len(self.entity_names)) * new_var
        temp_p[:old_n, :old_n] = self.mat_p
        self.mat_p = temp_p.copy()

        # TODO: Deal with mat_q
        self.mat_q = np.eye(len(self.entity_names))

    # TODO: Make compatible with lower level class ConnectivityUnit.update_new_tick
    def update_new_tick_conn_data(self, df_conn, measurement=None, mat_h=None, mat_r=None, keep_unit=False, relief_factor=0.6,
                                  forget_factor=0.8, **kwargs):
        """
        Update the risk estimates according to previous estimate

        :param df_conn: Canonical connection data DataFrame
        :param measurement: Risk measurements array
        :param mat_h: Observation matrix
        :param mat_r: Measurement noise covariance matrix
        :param keep_unit: If true stores the latest ConnectivityUnit as self.cu
        :param relief_factor: Percentage of the risk relieved at each time step for each node (see the paper for more)
        :param forget_factor: Linear interpolation hyperparameter that controls how much previous state graph should
        contribute to the current graph. 1 results in no contribution, 0 results in same state graphs. See the paper for
        more description
        :param kwargs: ConnectivityUnit.read_flows kwargs
        :return: None
        """

        df = preprocess_df(df_conn, date_col=' Timestamp')
        cu = ConnectivityUnit()

        cu.read_flows(df, entity_names=self.entity_names, **kwargs)
        cu.fit_connectivity_model(method='cov', verbose=True)

        if keep_unit:  # Only for debugging
            self.cu = cu

        self.mat_f = self.mat_f * (1 - forget_factor) + cu.mat_f * forget_factor
        self.mat_x, self.mat_p = single_risk_update(self.mat_f, measurement=measurement, mat_h=mat_h,
                                                    mat_x_init=self.mat_x,
                                                    mat_p_init=self.mat_p, mat_q=self.mat_q, mat_r=mat_r, k_steps=1,
                                                    relief_factor=relief_factor, normalize=False)

    def update_new_tick_samples(self, samples, names, measurement=None, mat_h=None, mat_r=None, keep_unit=False,
                                relief_factor=0.6, forget_factor=0.8):
        """
        Update the risk estimates according to previous estimate for the case samples are precomputed from connection
        data

        :param samples: Computed samples/observations of each entity. Rows are observation for enumerated entities
        :param names: Names of entities
        :param measurement: Risk measurements array
        :param mat_h: Observation matrix
        :param mat_r: Measurement noise covariance matrix
        :param keep_unit: If true stores the latest ConnectivityUnit as self.cu
        :param relief_factor: Percentage of the risk relieved at each time step for each node (see the paper for more)
        :param forget_factor: Linear interpolation hyperparameter that controls how much previous state graph should
        contribute to the current graph. 1 results in no contribution, 0 results in same state graphs. See the paper for
        more description
        :return: None
        """

        cu = ConnectivityUnit()
        cu.samples = samples
        cu.names = names
        cu.fit_connectivity_model(method='cov', verbose=True)

        if keep_unit:  # Only for debugging
            self.cu = cu

        self.mat_f = self.mat_f * (1 - forget_factor) + cu.mat_f * forget_factor
        self.mat_x, self.mat_p = single_risk_update(self.mat_f, measurement=measurement, mat_h=mat_h,
                                                    mat_x_init=self.mat_x,
                                                    mat_p_init=self.mat_p, mat_q=self.mat_q, mat_r=mat_r, k_steps=1,
                                                    relief_factor=relief_factor, normalize=False)

    def partition_network(self):
        # TODO: """ Partitions the network into subnetworks for simplicity"""
        pass
