import numpy as np
import pandas as pd
import stellargraph as sg
#

def graph_arrays_to_stellargraphs_labels(all_graphs, names, labels, thr=0):
    """ Makes a stellar graph out of the weight array representing a graph """
    all_graphs[np.abs(all_graphs) < thr] = 0

    graphs = []
    for k in range(all_graphs.shape[0]):
        weight_array = all_graphs[k, :, :]
        edge_df = pd.DataFrame(columns=[])
        for i, src in enumerate(names):
            for j, dst in enumerate(names):
                if j < i:
                    continue
                temp_df = pd.DataFrame({'source': src, 'target': dst, 'weight': weight_array[i, j]}, index=[0])
                edge_df = pd.concat((edge_df, temp_df), ignore_index=True)
        graphs.append(sg.StellarGraph(edges=edge_df))

    graph_labels = pd.DataFrame({'label': labels})
    graph_labels[graph_labels != 'BENIGN'] = 'ATTACK'
    return graphs, graph_labels
