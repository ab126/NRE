import networkx as nx

#  TODO: Fix warnings
def get_Graph(df, cls, ind = [1, 3, 6, 8, 9, 84]): #v2
    """
    Returns the directed graph for the respective dataframe. ind
    The digraph stands for the Communication graph of the network obtained
    from Sorce and Destination IP addresses of the connections"""
    G = nx.MultiDiGraph()
    G = nx.from_pandas_edgelist(df, cls[ind[0]], cls[ind[1]], list(cls[ind[3:5]]), create_using=nx.DiGraph())
    return G

def get_uGraph(df, cls, ind = [1, 3, 6, 8, 9, 84]): #v2
    """
    Returns the directed graph for the respective dataframe. ind
    The digraph stands for the Communication graph of the network obtained
    from Sorce and Destination IP addresses of the connections"""
    G = nx.MultiGraph()
    G = nx.from_pandas_edgelist(df, cls[ind[0]], cls[ind[1]], list(cls[ind[3:5]]), create_using=nx.Graph())
    return G

def get_data(G):
    """ Forms the data dictionary from Graph.
    Keys of data are the graph features"""
    btw_cnt = nx.betweenness_centrality(G)
    eig_cnt = nx.eigenvector_centrality(G, max_iter=10000)

    data = {}
    data['indegree'] = []
    data['outdegree'] = []
    data['betweenness centrality'] = []
    data['eigenvector centrality'] = []
    data['indeg_weight'] = []
    data['outdeg_weight'] = []
    data['clustering_coeff'] = []

    for node in G.nodes:
        G.nodes[node]['indeg_weight'] = 0
        G.nodes[node]['outdeg_weight'] = 0

    for u, v in G.edges:
        G.nodes[u]['outdeg_weight'] += int(G[u][v][' Total Fwd Packets'])
        G.nodes[u]['indeg_weight'] += int(G[u][v][' Total Backward Packets'])

        G.nodes[v]['indeg_weight'] += int(G[u][v][' Total Fwd Packets'])
        G.nodes[v]['outdeg_weight'] += int(G[u][v][' Total Backward Packets'])

    for node in G.nodes:
        G.nodes[node]['indegree'] = G.in_degree(node)
        G.nodes[node]['outdegree'] = G.out_degree(node)
        G.nodes[node]['betweenness centrality'] = btw_cnt[node]
        G.nodes[node]['eigenvector centrality'] = eig_cnt[node]
        G.nodes[node]['clustering_coeff'] = nx.clustering(G, node)

        data['indegree'].append(G.in_degree(node))
        data['outdegree'].append(G.out_degree(node))
        data['betweenness centrality'].append(btw_cnt[node])
        data['eigenvector centrality'].append(eig_cnt[node])
        data['indeg_weight'].append(G.nodes[node]['indeg_weight'])
        data['outdeg_weight'].append(G.nodes[node]['outdeg_weight'])
        data['clustering_coeff'].append(G.nodes[node]['clustering_coeff'])
    return data, G