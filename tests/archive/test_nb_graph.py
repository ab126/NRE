import networkx as nx
import numpy as np

from src.archive.causal_utils import generate_random_dag, add_linear_model, generate_samples


def generate_random_causal_linear_model_samples(n_node=5, n_sample=100, p_er_coef=0.75):
    """Generates a random causal linear model of n_node entities and samples from them"""
    p = p_er_coef * np.log(n_node) / n_node
    g_true = generate_random_dag(n_node, p)
    add_linear_model(g_true)
    exogenous_mean = [0 for _ in g_true.nodes]
    exogenous_var = [1 for _ in g_true.nodes]
    data_df = generate_samples(g_true, n_sample, exogenous_mean, exogenous_var)
    data_df = data_df[np.sort(data_df.columns)]
    mat_a = nx.to_numpy_array(g_true)
    return data_df, mat_a
