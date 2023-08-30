# Module for simulation utility functions
from collections import deque
import numpy as np
import numpy.random as nr
import networkx as nx
import pandas as pd
# from causallearn.search.ConstraintBased.PC import pc
# from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from scipy.stats import hmean, sem
from sklearn.linear_model import LinearRegression


def generate_random_dag(n, p, seed=None, keep_giant=False):
    """Generates a random dag based on ER random DiGraph by randomly removing edges
    from cycles. Keep giant = True for keeping only the giant component"""
    if not seed:
        nr.seed(seed)

    if seed:
        g = nx.erdos_renyi_graph(n, p, seed=seed, directed=True)
    else:
        g = nx.erdos_renyi_graph(n, p, directed=True)

    if keep_giant:
        if not nx.is_weakly_connected(g):  # Might reduce the number of nodes
            giant_comp = sorted(nx.weakly_connected_components(g), key=len, reverse=True)[0]
            g = g.subgraph(giant_comp)
            g = nx.relabel_nodes(g, {node: i for i, node in enumerate(g.nodes)})

    # Check for cycles and make dag
    while not nx.is_directed_acyclic_graph(g):
        cycles = list(nx.simple_cycles(g))
        k = nr.choice(np.arange(len(cycles)))
        cycle = cycles[k]
        i = nr.choice(np.arange(len(cycle)))
        j = i + 1 if i + 1 < len(cycle) else 0
        g.remove_edge(cycle[i], cycle[j])
    return g


def add_linear_model(g, seed=None, neg_weights=True):
    """Adds weights specifying the linear model about variables in dag g"""
    if not seed:
        nr.seed(seed)
    for (u, v) in g.edges():
        if not neg_weights:
            g.edges[u, v]['weight'] = nr.random()
        else:
            g.edges[u, v]['weight'] = (nr.random() - 0.5)


# Need an algorithm for sampling
def find_exogenous_nodes(g_rev):
    """Returns the nodes that have no edge coming at them from the reverse 
    representation of graph g"""
    exog_nodes = []
    for u in g_rev.nodes:
        if len(g_rev[u]) == 0:
            exog_nodes.append(u)
    return exog_nodes


def get_sampling_order(G):
    """Given a connected DAG returns orders to sample the nodes from according to dag"""
    # Find exogenous nodes
    g_rev = G.reverse()
    sampling_order = {}
    d = 0
    while g_rev.number_of_nodes() > 0:
        exogenous_nodes = find_exogenous_nodes(g_rev)
        for node in exogenous_nodes:
            sampling_order[node] = d
        g_rev.remove_nodes_from(exogenous_nodes)
        d += 1

    return sampling_order


def get_key(my_dict, target):
    """Return keys with target value"""
    keys = []
    for key, val in my_dict.items():
        if val == target:
            keys.append(key)
    return keys


def generate_samples(g, n_samp, exogenous_mean, exogenous_var):
    """
    Samples from the weighted dag g according to linear gaussian model with coefficients
    as edge weights and exogenous variables given as input"""
    sampling_order = get_sampling_order(g)
    assert -1 not in sampling_order.values(), "Orders are not set properly!"
    g_rev = g.reverse()

    data_df = pd.DataFrame()
    for d in np.sort(list(sampling_order.values())):
        nodes = get_key(sampling_order, d)
        for node in nodes:
            parents = list(g_rev[node].keys())
            samples = nr.normal(exogenous_mean[node], exogenous_var[node], n_samp)
            if parents is not None:
                for par in parents:
                    samples += g[par][node]['weight'] * data_df[par]
            data_df[node] = samples
    return data_df


def get_perf_scores(G_pc, G_true, default_tpr=1):
    """Given the NetworkX graph returned by the pc algorithm and the true graph returns the performance metrics"""

    wk_acc = wk_fpr = wk_tpr = 0
    st_acc = st_fpr = st_tpr = 0
    for u in G_true.nodes:
        for v in G_true.nodes:
            if u == v:
                continue
            if v in G_true[u]:
                if v in G_pc[u]:  # True Positive
                    wk_acc += 1
                    wk_tpr += 1
                    if G_pc[u][v]['color'] in ['b', 'r']:
                        st_acc += 1
                        st_tpr += 1
                elif u in G_pc[v]:  # Other way around edge exists
                    wk_acc += 1
                    wk_tpr += 1
                else:  # False Negative
                    pass
            else:  # v not in g_true[u]
                if v in G_pc[u]:  # False Positive
                    if u in G_true[v]:  # Other way around edge exists
                        wk_acc += 1
                    else:
                        wk_fpr += 1
                    st_fpr += 1
                else:  # True Negative
                    wk_acc += 1
                    st_acc += 1

    tot_poss_edges = len(G_true.nodes) * (len(G_true.nodes) - 1)
    tot_edges = len(G_true.edges)

    wk_acc /= tot_poss_edges
    st_acc /= tot_poss_edges
    if tot_edges != 0:
        wk_tpr /= tot_edges
        st_tpr /= tot_edges
    else:
        assert wk_tpr == 0, "No edges in g_true but Weak TPR is non-zero"
        assert st_tpr == 0, "No edges in g_true but Strong TPR is non-zero"
        wk_tpr = st_tpr = default_tpr

    wk_fpr /= tot_poss_edges - tot_edges
    st_fpr /= tot_poss_edges - tot_edges

    return wk_acc, wk_tpr, wk_fpr, st_acc, st_tpr, st_fpr


def get_lin_model_stats(G, magnitude=True):
    """Returns some stats about linear model embedded in dag g"""
    coefficients = []
    for u in G.nodes:
        for v in G[u]:
            coefficients.append(G[u][v]['weight'])
    if len(coefficients) == 0:
        return 0, 0, 0, 0
    if magnitude:
        coefficients = np.abs(coefficients)
    return hmean(coefficients), np.mean(coefficients), np.min(coefficients), np.max(coefficients)


def single_run_results(n_node=5, n_samp=1000, p=None, p_er_coef=0.75, run_num=0,
                       exogenous_mean=None, exogenous_var=None, default_tpr=1, forb_edge_perc=0):
    """Gets the Single run results with given experiment parameters"""

    if p is None:
        p = p_er_coef * np.log(n_node) / n_node
    else:
        p_er_coef = p * n_node / np.log(n_node)

    g_true = generate_random_dag(n_node, p)
    add_linear_model(g_true)
    if not exogenous_mean:
        exogenous_mean = [0 for _ in g_true.nodes]
    if not exogenous_var:
        exogenous_var = [1 for _ in g_true.nodes]

    data_df = generate_samples(g_true, n_samp, exogenous_mean, exogenous_var)
    data_df = data_df[np.sort(data_df.columns)]

    if forb_edge_perc == 0:
        cg = pc(data_df.values, node_names=data_df.columns, show_progress=False)
    else:
        bk = make_forbidden_edges_bk(forb_edge_perc, g_true)
        cg = pc(data_df.values, node_names=data_df.columns,
                background_knowledge=bk, show_progress=False)

    cg.to_nx_graph()
    g_pc = cg.nx_graph.copy()
    wk_acc, wk_tpr, wk_fpr, st_acc, st_tpr, st_fpr = get_perf_scores(g_pc, g_true, default_tpr=default_tpr)
    coef_hmean, coef_mean, coef_min, coef_max = get_lin_model_stats(g_true)

    single_run_df = pd.DataFrame(
        {'Run #': run_num, 'Sample Size': n_samp, 'DAG Size': n_node, 'Edge Prob.': p, 'ER Prob. Coef.': p_er_coef,
         'Weak Accuracy': wk_acc, 'Weak TPR': wk_tpr, 'Weak FPR': wk_fpr,
         'Strong Accuracy': st_acc, 'Strong TPR': st_tpr, 'Strong FPR': st_fpr,
         'Coef. Mean': coef_mean, 'Coef. H. Mean': coef_hmean, 'Coef. Min': coef_min,
         'Coef. Max': coef_max, 'Forbidden Edge %': forb_edge_perc}, index=[0])
    return single_run_df


def simulate_over_param_space(n_runs=50, n_nodes=[5], c_p_ERs=[0.75], n_samples=[1000],
                              forb_edge_perc=[0]):
    """
    Simulates the parameter space. Cartesian product of parameter arrays are simulated

    n_runs : Number of times experiment is repeated with new rando DAG
    n_nodes : Array of number of variables/ nodes in DAG
    c_p_ERs : Constants in p = c * np.log(n_node)/ n_node for edge probablity for Random DAGs
    n_samples : Number of samples taken from linear model about variable
    
    Returns : Result of simulation as pandas.DataFrame
    """
    res_df = pd.DataFrame()
    for i in range(n_runs):
        print(i)
        for n_samp in n_samples:
            for n_node in n_nodes:
                for c in c_p_ERs:
                    for perc in forb_edge_perc:
                        sim_kwargs = {'n_node': n_node, 'n_samp': n_samp, 'p_er_coeff': c, 'run_num': i,
                                      'forb_edge_perc': perc}
                        temp_df = single_run_results(**sim_kwargs, default_tpr=0)

                        # Add graph features Here
                        res_df = pd.concat((res_df, temp_df), ignore_index=True)
    return res_df


def get_r_squared(res_df, x_var, y_var):
    """Returns the Coefficient of Determination from simulation results for predicting y_var from x_Var"""
    mat_x, y = res_df[x_var].values.reshape((-1, 1)), res_df[y_var].values
    model = LinearRegression()
    model.fit(mat_x, y)
    return model.score(mat_x, y)


def boostrap_r_squared(res_df, x_var, y_var, n_bootstrap=1000):
    """Uses Bootstrapping to get mean R^2 estimate and confidence intervals"""
    n_rows = res_df.shape[0]
    r_2_scores = []
    for _ in range(n_bootstrap):
        ind = nr.choice(np.arange(n_rows), n_rows)
        temp_df = res_df.iloc[ind, :].copy()
        r_2_scores.append(get_r_squared(temp_df, x_var, y_var))
    return r_2_scores


def get_se_conf_int(vals, alpha=0.05):
    """Given statistics vals returns mean, standard error and two-sided confidence intervals"""
    val_mean = np.mean(vals)
    val_se = sem(vals)
    n = len(vals)
    sortd_vals = np.sort(vals)
    low = int(np.floor(n * alpha / 2))
    high = int(np.ceil(n - n * alpha / 2))
    conf_int = sortd_vals[low], sortd_vals[high]
    return val_mean, val_se, conf_int


def get_forbidden_edges(n_perc, g_true):
    """Returns random n_perc% edges that are not in g_true"""
    assert 1 >= n_perc >= 0, "n_perc must be in [0, 1] interval"
    forbidden_edges = list(nx.complement(g_true).edges())
    ind = nr.permutation(np.arange(len(forbidden_edges)))
    end_val = int(n_perc * len(forbidden_edges))
    return (np.array(forbidden_edges)[ind[:end_val]]).tolist()


def make_forbidden_edges_bk(n_perc, g_true):
    """Returns a BackgroundKnowledge object for the given percentage of bk edges"""
    # Generate some data_df to get causal graph node objects
    exogenous_mean = [0 for node in g_true.nodes]
    exogenous_var = [1 for node in g_true.nodes]
    data_df = generate_samples(g_true, 10 * len(g_true), exogenous_mean, exogenous_var)
    data_df = data_df[np.sort(data_df.columns)]
    cg0 = pc(data_df.values, node_names=data_df.columns, show_progress=False)
    cg_nodes = cg0.G.get_nodes()
    cg_node_dict = {node: cg_node for node, cg_node in zip(data_df.columns, cg_nodes)}

    forb_edges = get_forbidden_edges(n_perc, g_true)
    bk = BackgroundKnowledge()
    for u, v in forb_edges:
        bk.add_forbidden_by_node(cg_node_dict[u], cg_node_dict[v])

    return bk
