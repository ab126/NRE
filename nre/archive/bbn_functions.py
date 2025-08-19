import networkx as nx
import random
import numpy as np
from scipy.linalg import norm
from pomegranate import *


def binary_n_values(n):
    "Returns the list of all outcomes n binary random variables can produce"

    if n == 0:
        return [[]]

    prev_vals = binary_n_values(n - 1)
    n_vals = []

    for i, el_list in enumerate(prev_vals):
        temp_list0 = ['0']
        temp_list0.extend(el_list)
        n_vals.append(temp_list0)

    for i, el_list in enumerate(prev_vals):
        temp_list1 = ['1']
        temp_list1.extend(el_list)
        n_vals.append(temp_list1)

    return n_vals


def get_rand_CPT_list(num_par):
    "Returns the random Conditional Probability Table for nodes having parents"

    assert num_par >= 1

    cond_var_vals = binary_n_values(num_par)
    CPT_list = []

    for i, el_list in enumerate(cond_var_vals):
        rand_p = random.random()
        temp_list = el_list.copy()
        temp_list.extend(['0', rand_p])
        CPT_list.append(temp_list)

        temp_list = el_list.copy()
        temp_list.extend(['1', 1 - rand_p])
        CPT_list.append(temp_list)

    return CPT_list


def generate_BBN_model(DAG):
    """
    Generates the model with random probability tables given for all the nodes
    in the BNN
    """
    rev_DAG = DAG.reverse(copy=True)
    state_list = []
    state_names = []

    nodes = np.array(DAG.nodes)
    nodes.sort()
    for node in nodes:
        num_par = len(rev_DAG[node])
        if num_par == 0:
            num = random.random()
            p_dist = DiscreteDistribution({'0': num, '1': 1 - num})

        else:
            CPT_list = get_rand_CPT_list(num_par)
            parent_list = list(rev_DAG[node].keys())
            par_state_list = [DAG.nodes[parent]['p_dist'] for parent in parent_list]
            p_dist = ConditionalProbabilityTable(CPT_list, par_state_list)
        DAG.nodes[node]['p_dist'] = p_dist
        state = State(p_dist, name=str(node))
        state_list.append(state)
        state_names.append(str(node))

    state_dict = {name: val for name, val in zip(state_names, state_list)}

    model = BayesianNetwork("Viral Transmission Problem")
    for name in state_names:
        model.add_state(state_dict[name])

    # Edges
    for n1, n2 in DAG.edges:
        model.add_edge(state_dict[str(n1)], state_dict[str(n2)])
    model.bake()
    return model, state_dict, state_names


def kl_div(p, q):
    "Calculates kl divergence of two layed-out probability distributions"
    return np.sum(np.multiply(p, np.log2(np.divide(p, q))))


def get_DAG_from_model(model):
    "Returns the Directed Acyclic Graph from Bayesian Network model"
    edge_list = [(s1.name, s2.name) for s1, s2 in model.edges]
    G = nx.DiGraph()
    G.add_edges_from(edge_list)

    # Adding lone nodes
    state_names = [state.name for state in model.states]
    for node in state_names:
        if node not in G.nodes:
            G.add_node(node)
    return G.copy()


def frobenious_graph_error(m0, m1, err_str='relative'):
    """Frobenious norm based graph distance of two Bayesian Network models"""
    G0 = get_DAG_from_model(m0)
    G1 = get_DAG_from_model(m1)

    A0 = nx.adjacency_matrix(G0)
    A1 = nx.adjacency_matrix(G1)

    Ad = A0 - A1
    if err_str == 'absolute':
        return norm(Ad.todense())
    else:
        a0 = np.array(A0.todense()).flatten()
        a1 = np.array(A1.todense()).flatten()

        cos_theta = np.inner(a0, a1) / np.sqrt(np.inner(a0, a0) * np.inner(a1, a1))
        return 1 - cos_theta


def kl_div2(p, q):
    """Calculates kl divergence of two layed-out probability distributions"""
    div = 0
    for i in range(len(p)):
        if q[i] == 0 or p[i] == 0:
            continue
        div += p[i] * np.log2(p[i] / q[i])
    return div


# Safe Random generation functions

def get_next_vals(prev_list):
    "Given the list of binary values ('0' or '1') returns lists that have 1 more '1' than input"
    next_lists = []
    for i, val in enumerate(prev_list):
        if val == '0':
            temp = prev_list.copy()
            temp[i] = '1'
            next_lists.append(temp.copy())
    return next_lists.copy()


def get_distant_p(p):
    "Given a probability p returns a distant probability from p"
    delta = 0.5
    mean = np.mod(p + delta, 1)
    width = 0.2
    return np.mod(random.uniform(mean - width / 2, mean + width / 2), 1)


def add_consistent_rows2CPT_list(CPT_list, el_list, rand_p):
    "Adds consistent lines to the CPT_list reflecting the node of interest's probability given its parents"
    temp_list = el_list.copy()
    temp_list.extend(['0', rand_p])
    CPT_list.append(temp_list)

    temp_list = el_list.copy()
    temp_list.extend(['1', 1 - rand_p])
    CPT_list.append(temp_list)


def get_safe_rand_CPT_list(num_par):
    "Returns the random Conditional Probability Table for nodes having parents"

    assert num_par >= 1
    curr_vals_list = [['0' for i in range(num_par)]]
    CPT_list = []
    curr_p = random.random()

    for n_1s in range(num_par + 1):
        for curr_vals in curr_vals_list:
            add_consistent_rows2CPT_list(CPT_list, curr_vals, curr_p)

        curr_p = get_distant_p(curr_p)
        curr_vals_list = get_next_vals(curr_vals)

    return CPT_list


def generate_safe_BBN_model(DAG):
    """
    Generates the model with random probability tables given for all the nodes
    in the BNN
    """
    rev_DAG = DAG.reverse(copy=True)
    state_list = []
    state_names = []

    nodes = np.array(DAG.nodes)
    nodes.sort()
    for node in nodes:
        num_par = len(rev_DAG[node])
        if num_par == 0:
            num = random.random()
            p_dist = DiscreteDistribution({'0': num, '1': 1 - num})

        else:
            CPT_list = get_safe_rand_CPT_list(num_par)
            # print(CPT_list)
            parent_list = np.array(list(rev_DAG[node].keys()))
            parent_list.sort()  # ?
            par_state_list = [DAG.nodes[parent]['p_dist'] for parent in parent_list]
            p_dist = ConditionalProbabilityTable(CPT_list, par_state_list)
        DAG.nodes[node]['p_dist'] = p_dist
        state = State(p_dist, name=str(node))
        state_list.append(state)
        state_names.append(str(node))

    state_dict = {name: val for name, val in zip(state_names, state_list)}

    model = BayesianNetwork("Viral Transmission Problem")
    for name in state_names:
        model.add_state(state_dict[name])

    # Edges
    for n1, n2 in DAG.edges:
        model.add_edge(state_dict[str(n1)], state_dict[str(n2)])
    print(model)
    model.bake()
    return model, state_dict, state_names
