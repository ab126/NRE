import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from sklearn.cluster import SpectralClustering


def get_eig_vals_err(mat_f, node_names, plot_bool=True):
    """
    Returns sorted (ascending) eigenvalues of the Laplacian and reconstruction error
    """
    g = nx.from_numpy_array(mat_f, create_using=nx.DiGraph)
    g = g.to_undirected()  # should make it undirected first
    g = nx.relabel_nodes(g, {node_names.index(node): node for node in node_names})
    mat_l_sym = nx.normalized_laplacian_matrix(g)
    mat_l_sym = mat_l_sym.todense()

    eigvals, eigvecs = np.linalg.eig(mat_l_sym)

    eigvals = np.real(eigvals)
    sort_ind = np.argsort(eigvals)
    eigvals = eigvals[sort_ind]
    eigvecs = eigvecs[:, sort_ind]

    err = np.linalg.norm(mat_l_sym - eigvecs * np.diag(eigvals) * eigvecs.T) / np.linalg.norm(mat_l_sym)
    if plot_bool:
        sns.scatterplot(x=np.arange(eigvals.shape[0]), y=eigvals)
        plt.ylabel('Eigenvalues', fontsize=14)
        plt.title('Eigenvalues of Graph Laplacian', fontsize=14)
        plt.show()
    return eigvals, err


def apply_spec_clus(mat_f, node_names, n_clus, plot_bool=True, fontsize=24, seed=5):
    """
    Applies spectral clustering to nx graph g returns the assigned clusters

    :param mat_f: Numpy 2d array representing the adjacency matrix
    :param node_names: List of names according to mat_f
    :param n_clus: Number of clusters
    :param plot_bool: If True, displays the clusters overlayed on top of mat_f
    :param fontsize: Fontsize used in plotting
    :param seed: Seed used for spectral clustering
    :return gr, all_labels, all_clusters:
        gr: Original graph reordered according to clusters
        all_labels: Cluster labels of all nodes
        all_clusters: Indices of all clusters
    """

    g = nx.from_numpy_array(mat_f, create_using=nx.DiGraph)
    g = g.to_undirected()  # should make it undirected first
    g = nx.relabel_nodes(g, {node_names.index(node): node for node in node_names})

    # Break into connected components
    gcc = sorted(nx.connected_components(g), key=len, reverse=True)
    cur_label = 0  # Separate the labels of connected components
    all_nodes = np.array([])
    all_labels = np.array([])
    for sub_nodes_set in gcc:
        g_sub = g.subgraph(sub_nodes_set)
        if len(g_sub) == 1:
            all_nodes = np.concatenate((all_nodes, list(sub_nodes_set)))
            all_labels = np.concatenate((all_labels, np.array([cur_label]).astype(int)))
            cur_label += 1
            continue
        mat_w = nx.adjacency_matrix(g_sub).todense()
        clustering = SpectralClustering(n_clusters=n_clus, affinity='precomputed', assign_labels='discretize',
                                        random_state=seed).fit(mat_w)
        # Reordering Nodes
        clusters = np.unique(clustering.labels_)
        node_idx = []
        for cluster in clusters:
            indices = np.where(clustering.labels_ == cluster)[0]
            node_idx.extend(list(indices))
        nodes = list(np.array(list(sub_nodes_set))[node_idx])

        # Add to the ultimate lists
        all_nodes = np.concatenate((all_nodes, nodes))
        all_labels = np.concatenate((all_labels, (clustering.labels_[node_idx] + cur_label).astype(int)))
        cur_label += len(clusters)
    all_labels = all_labels.astype(int)
    all_clusters = np.unique(all_labels)

    # Reordered g
    g_relabeled = nx.Graph()
    g_relabeled.add_nodes_from(all_nodes)
    g_data = [(u, v, d['weight']) for (u, v, d) in g.edges(data=True)]
    g_relabeled.add_weighted_edges_from(g_data)

    if plot_bool:
        plt.rcParams.update({'font.size': fontsize})
        # Rendering Clusters
        plt.figure(figsize=(10, 10))
        plt.imshow(nx.adjacency_matrix(g_relabeled).todense(), interpolation='nearest') # cmap='Blues'
        plt.colorbar()
        cm_strs = ['Reds', 'Reds']
        for i, cluster in enumerate(all_clusters):
            temp = all_labels == cluster
            label_mask = np.outer(temp, temp)
            masked_array = np.ma.masked_where(label_mask == False, label_mask)
            cmap = matplotlib.cm.get_cmap('Reds')
            cmap.set_bad(color='#FF000000')
            plt.imshow(masked_array, cmap=cmap, alpha=.5)
        plt.show()
        plt.rcParams.update({'font.size': 10})
    return g_relabeled, all_labels, all_clusters


def get_graph_clus(gr, new_labels, clusters, i, plot_bool=True, fontsize=18):
    """Given reordered clustered graph gr returns the ith cluster (starting from 0)"""

    nodes = np.array(gr.nodes)
    ind = new_labels == clusters[i]
    h = gr.subgraph(nodes[ind]).copy()  # Subgraph

    # Reorder
    hrr = nx.Graph()
    hrr.add_nodes_from(nodes[ind])
    g_data = [(u, v, det['weight']) for (u, v, det) in h.edges(data=True)]
    hrr.add_weighted_edges_from(g_data)
    h = hrr.copy()

    if plot_bool:
        old_indices = np.arange(len(gr))[ind]
        plt.rcParams.update({'font.size': fontsize})
        mat_a = nx.adjacency_matrix(h).todense()
        cond_num = np.linalg.cond(mat_a)
        det = np.linalg.det(mat_a.T * mat_a)
        print('Cluster ' + str(i) + '\nConditioning number: ', cond_num, '\nDeterminant of F^T*F: ', det)
        plt.matshow(mat_a, vmin=0, extent=[old_indices[0], old_indices[-1], old_indices[-1], old_indices[0]])
        ax = plt.gca()
        ax.xaxis.set_ticks_position('bottom')
        plt.xticks(rotation=45)
        plt.colorbar()
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)
        plt.show()
        plt.rcParams.update({'font.size': 10})
    return h


def get_clus_assign(s_nodes, gr, new_labels):
    """Returns the cluster assignments from partitioned graph gr given the nodes"""
    clus_assignment = {}
    for node in s_nodes:
        if node not in gr.nodes:
            clus_assignment[node] = -1
        else:
            clus_assignment[node] = new_labels[list(gr.nodes).index(node)]
    return clus_assignment
