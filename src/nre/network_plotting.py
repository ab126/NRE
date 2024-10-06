import matplotlib
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from networkx.drawing.layout import rescale_layout

from .network_partitioning import apply_spec_clus


# Safe Routing Plotting Functions
def polar2cartesian(r, theta, units='deg'):
    """Return the cartesian coordinates (x, y) corresponding to the polar coordinates"""
    if units == 'deg':
        theta = theta * np.pi / 180
    else:
        assert units == 'rad', "Implemented for units 'deg' or 'rad'"
    return r * np.cos(theta), r * np.sin(theta)


def cartesian_dist_plot(g, distances, risks, destination=None, title='', dx=1, dy=1, n_points=1000, plot=True, y_max=3,
                        show_names=True, label_size=8, paths_highlight=None, discrete_color=False):
    """
    Depicts the network according to the calculated distances to route from source to every other entity. Fits the
    entities to integer cartesian coordinates where the source is left most and every other entity is distributed according
    to their distance.

    :param g: NetworkX graph which stands for the communication graph of the network
    :param distances: Dictionary of distances or risk-levels from the source. The Dictionary is ordered with increasing
        distances
    :param risks: Dictionary of entity risks
    :param destination: (optional) Destination node name
    :param title: Title of the plot
    :param dx: Increment in x direction
    :param dy: Increment in y direction
    :param plot: If True, show the plot
    :param n_points: Number of points to draw vertical lines
    :param show_names: If True, shows the names of entities as well.
    :param label_size: Fontsize for labels.
    :param paths_highlight: List of paths, that are list of node tuples, to highlight.
    :param discrete_color: If True, colors the entities uniformly separated for better visual
    :return fig: Figure of the network plot
    """
    source = list(distances.keys())[0]
    vals, indices, counts = np.unique(list(distances.values()), return_inverse=True, return_counts=True)

    x_cur, y_cur = 0, 0
    idx_prev = -1
    fig, ax = plt.subplots(figsize=(9, 6))
    pos = {}
    max_num = np.max(counts)
    x_layers = []
    for i, node in enumerate(distances.keys()):
        idx = indices[i]
        num = counts[indices[i]]  # Number of same distance entities
        ddy = 1 if num == 1 else y_max * 2 / (num - 1)

        # Update coordinates
        if idx != idx_prev:  # New layer
            if num != 1:
                # y_init = - (num - 1) / 2 * dy
                y_init = -y_max
            elif i == 0 or i == len(distances) - 1:  # First or last entity
                y_init = 0
            else:  # Single entity in middle layers
                # y_init = (-1) ** (i + 1) * dy / 2
                y_init = (-1) ** (i + 1) * y_max / 2
            x_cur, y_cur = x_cur + dx, y_init
            x_layers.append(x_cur)
        else:
            # x_cur, y_cur = x_cur, y_cur + dy
            x_cur, y_cur = x_cur, y_cur + ddy
        pos[node] = np.array([x_cur, y_cur])
        idx_prev = idx

    # Plot lines
    ys = np.linspace(-(max_num / 2 + 0.5), max_num / 2 + 0.5, n_points) * dy
    for x_cur in x_layers:
        xs = np.array([x_cur for _ in ys])
        ax.plot(xs, ys, 'tab:gray', alpha=.2)

    # Plot x axis
    y_x_ax = -(max_num / 2 + 1)
    ax.plot([x_layers[0], x_layers[-1]], [y_x_ax, y_x_ax], color="k")
    ax.plot(x_layers[-1], y_x_ax, ls="", marker=">", ms=5, color="k", clip_on=False)

    # Plot the rest
    cmap = plt.cm.YlOrRd  # plt.cm.Reds
    node_clr = [risks[node] for node in g.nodes]
    if discrete_color:
        lin_color = np.linspace(min(node_clr), max(node_clr), len(node_clr))
        nodes = list(g.nodes)
        risks = {nodes[ind]: lin_color[i] for i, ind in enumerate(np.argsort(node_clr))}

    norm = matplotlib.colors.Normalize(vmin=min(node_clr), vmax=max(node_clr))

    # Draw most nodes
    most_nodes = [node for node in g.nodes if node != source and node != destination]
    most_clr = [risks[node] for node in most_nodes]
    if show_names:
        pos_label = {node: pos[node] + (-.02, -0.5) for node in pos}
        labels = {node: '' + '.'.join(node.split('.')[-4:]) for node in pos}
        nx.draw_networkx_labels(g, pos_label, labels=labels, ax=ax, clip_on=False, font_size=label_size)
    most_nodes = nx.draw_networkx_nodes(g, pos, nodelist=most_nodes, cmap=cmap, node_color=most_clr, ax=ax)
    most_nodes.set_edgecolor('black')
    most_nodes.set_linewidth(2)
    nx.draw_networkx_edges(g, pos, width=1.2, edge_color='black', ax=ax)

    # Draw the source
    options = {"node_size": 350, "node_shape": 's', "node_color": cmap(norm(risks[source]))}
    src_node = nx.draw_networkx_nodes(g, pos, nodelist=[source], ax=ax, **options)
    src_node.set_edgecolor('black')
    src_node.set_linewidth(2)

    # Draw Destination
    if destination is not None:
        options = {"node_size": 350, "node_shape": 'D', "node_color": cmap(norm(risks[destination]))}
        dst_node = nx.draw_networkx_nodes(g, pos, nodelist=[destination], ax=ax, **options)
        dst_node.set_edgecolor('black')
        dst_node.set_linewidth(2)
        if paths_highlight is not None:
            clrs = np.tile(['tab:green', 'tab:red'], len(paths_highlight) // 2 + 1)
            for i, path in enumerate(paths_highlight):
                nx.draw_networkx_edges(g, pos, edgelist=path, width=3, edge_color=clrs[i], ax=ax,
                                       arrows=True, arrowstyle='-|>')

    ax.annotate('Source', xy=pos[source], xytext=(-1.6, 1.2), size=12, annotation_clip=False,
                textcoords='offset fontsize')
    ax.annotate('Destination', xy=pos[destination], xytext=(-2.8, 1.3), size=12, annotation_clip=False,
                textcoords='offset fontsize')

    cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.05, shrink=0.6)
    cbar.set_label('Mean Entity Risks', labelpad=-55)
    ax.set_title(title)
    plt.box(False)

    # Set x ticks
    ax.spines['top'].set_visible(True)
    ax.set_xlabel('Increasing Path Risk From the Source')
    ax.get_xaxis().set_ticks(ticks=x_layers)

    if plot:
        plt.show()
    return fig


def polar_dist_plot(g, distances, risks, title='', plot=True, n_points=1000, seed=66):
    """
    Depicts the network according to the calculated distances to route from source to every other entity. Distributes
    the entities around the source with respect to the distances.

    :param g: NetworkX graph which stands for the communication graph of the network
    :param distances: Dictionary of distances or risk-levels from the source. The Dictionary is ordered with increasing
        distances
    :param risks: Dictionary of entity risks
    :param title: Title of the plot
    :param plot: If True, show the plot
    :param n_points: Number of points to draw radial circles
    :param seed: Seed for the initial angle for different distant entities
    :return fig: Figure of the network plot
    """
    np.random.seed(seed)
    source = list(distances.keys())[0]
    thetas = np.linspace(0, 2 * np.pi, n_points)
    vals, indices, counts = np.unique(list(distances.values()), return_inverse=True, return_counts=True)

    # Distribute different distant entities
    n_dist = len(vals)
    theta_init = np.arange(n_dist) * 360 / n_dist
    theta_init = np.random.permutation(theta_init)
    r_init = np.sqrt(np.linspace(0, n_dist ** 2, n_points))

    r_cur, theta_cur = -1, -theta_init[0]
    idx_prev = -1
    fig, ax = plt.subplots()  # figsize=(15, 10))
    pos = {}
    for i, node in enumerate(distances.keys()):
        idx = indices[i]
        num = counts[indices[i]]  # Number of same distance entities
        dtheta = 360 / num

        # Update coordinates
        if idx != idx_prev:
            r_cur, theta_cur = r_init[indices[i]], theta_init[indices[i]]

            xs, ys = r_cur * np.cos(thetas), r_cur * np.sin(thetas)
            ax.plot(xs, ys, 'tab:gray', alpha=.2)
        else:
            theta_cur = theta_cur + dtheta

        x_cur, y_cur = polar2cartesian(r_cur, theta_cur)
        pos[node] = np.array([x_cur, y_cur])

        idx_prev = idx

    # Plotting
    cmap = plt.cm.Reds  # plt.cm.Reds
    node_clr = [risks[node] for node in g.nodes]
    norm = matplotlib.colors.Normalize(vmin=min(node_clr), vmax=max(node_clr))

    # nx.draw(g, pos=pos, with_labels=False, cmap=cmap, node_color=node_clr, width=0.2)
    most_nodes = nx.draw_networkx_nodes(g, pos, cmap=cmap, node_color=node_clr)
    most_nodes.set_edgecolor('black')
    most_nodes.set_linewidth(2)
    nx.draw_networkx_edges(g, pos, width=0.8, edge_color='darkred')

    # Draw the source
    options = {"node_size": 350, "node_shape": 's', "node_color": cmap(norm(risks[source]))}
    src_node = nx.draw_networkx_nodes(g, pos, nodelist=[source], **options)
    src_node.set_edgecolor('black')
    src_node.set_linewidth(2)

    ax.annotate('Source', pos[source] + (-0.065, 0.07), size=9)
    cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.05, shrink=0.6)
    cbar.set_label('Entity Risks', labelpad=-50)
    ax.set_title(title)
    plt.box(False)

    if plot:
        plt.show()
    return fig


def normalize_coordinates(pos, risk_mean=None, risk_cov=None, diam_xy=4.0, diam_z=2.0):
    """
    Normalizes the coordinates of the nodes in each axis according the diameter diam_xy

    :param pos: X-y coordinates of entities
    :param risk_mean: Mean of estimated risks are mapped to z coordinate in layout
    :param risk_cov: Covariance is scaled according to risk_mean
    :param diam_xy: Difference in max and min coordinates along x & y-axis (assumed same)
    :param diam_z: Difference in max and min coordinates along z axis
    :return pos, risk_mean, risk_cov: Returns the normalized input
    """

    # Read the diam
    max_x, max_y, min_x, min_y = -np.inf, -np.inf, np.inf, np.inf
    max_risk = 0

    # Get max coordinates
    for node in pos:
        x, y = pos[node]
        max_x, max_y, min_x, min_y = max(x, max_x), max(y, max_y), min(x, min_x), min(y, min_y)

    if risk_mean is not None:
        for val in risk_mean:
            max_risk = max(val, max_risk)

    # Adjust diam
    mult_x = diam_xy / (max_x - min_x) if max_x != min_x else 1
    mult_y = diam_xy / (max_y - min_y) if max_y != min_y else 1
    mult_z = diam_z / max_risk if risk_mean is not None else 1
    base_xy = np.array([min_x, min_y])
    for node in pos:
        pos[node] = np.multiply(pos[node] - base_xy, np.array([mult_x, mult_y])) - diam_xy / 2
    risk_mean = risk_mean * mult_z if risk_mean is not None else None
    risk_cov = risk_cov * (mult_z ** 2) if risk_mean is not None else None
    if risk_mean is not None:
        return pos, risk_mean, risk_cov
    else:
        return pos


def pos2json(filename, **kwargs):
    """
    Writes the {entity: np.array(., .)} dictionaries to the filename.json file

    :param filename: .json file name
    :param kwargs: {keys:vals} keys are added to the json object as field whose values are vals
    """
    json_dict = {}
    for key, val in kwargs.items():
        json_dict[key] = val
    with open(filename + '.json', "w") as outfile:
        json.dump(json_dict, outfile)


# TODO: Add method parameter
def pie_layout(mat_f, entity_names, n_cluster, risk_mean=None, risk_cov=None, d_xy=1.5, d_z=2, r_const=4, alpha=2,
               iterations=50, plot_bool=True, with_labels=False, idle=False):
    """Computes the "Pie Layout" for the network given by mat_f"""

    g_relabeled, new_labels, clusters = apply_spec_clus(mat_f, entity_names, n_cluster, plot_bool=False)
    clus_assgn = {node: int(i) for node, i in zip(g_relabeled.nodes, new_labels)}

    phi = 2 * np.pi / (n_cluster - 1) if n_cluster > 1 else 2 * np.pi  # Constant angle of pie given for each cluster
    r = r_const * d_xy / phi
    scale_cluster = 0.8
    cmap = plt.get_cmap('cool')

    pos_pie = {}
    node_colors = {}
    for i in clusters:

        ind = new_labels == i
        g_sub = g_relabeled.subgraph(np.array(g_relabeled.nodes)[ind])
        theta_i = phi * (i - 1)

        if idle:
            pos = nx.spring_layout(g_sub, seed=43)
        else:
            pos = risk_elevation_layout(g_sub, alpha=alpha, seed=43, iterations=iterations)

        if risk_mean is None:
            pos = normalize_coordinates(pos, diam_xy=d_xy, diam_z=d_z)
        else:
            # assert risk_cov is not None, "Risk_mean is not None, but Risk_cov is"
            pos, risk_mean, risk_cov = normalize_coordinates(pos, risk_mean=risk_mean, risk_cov=risk_cov, diam_xy=d_xy,
                                                             diam_z=d_z)

        for node in g_sub.nodes:
            x, y = pos[node] * scale_cluster
            if i == 0:
                pos_pie[node] = np.array([x, y])
            else:
                pos_pie[node] = np.array(polar2cartesian(r + y, np.pi / 2 - (x / d_xy * phi + theta_i), units='rad'))
            node_colors[node] = cmap(i / n_cluster)

    g_out = g_relabeled.subgraph(entity_names)  # Reordering nodes according to original naming
    if plot_bool:
        fig, ax = plt.subplots(figsize=(10, 6))
        widths = [g_out[u][v]['weight'] for u, v in g_out.edges]
        colors = [node_colors[name] for name in entity_names]
        nx.draw(g_out, pos_pie, with_labels=with_labels, width=widths, ax=ax, node_color=colors)
    if risk_mean is None:
        return pos_pie, node_colors, r, g_out, clus_assgn
    else:
        return pos_pie, risk_mean, risk_cov, node_colors, r, g_out, clus_assgn


# --------------------------------------------------------------------------------
# Taken and modified from NetworkX library. See
# https://networkx.org/documentation/stable/_modules/networkx/drawing/layout.html
# --------------------------------------------------------------------------------
def _process_params(G, center, dim):
    # Some boilerplate code.

    if not isinstance(G, nx.Graph):
        empty_graph = nx.Graph()
        empty_graph.add_nodes_from(G)
        G = empty_graph

    if center is None:
        center = np.zeros(dim)
    else:
        center = np.asarray(center)

    if len(center) != dim:
        msg = "length of center coordinates must match dimension of layout"
        raise ValueError(msg)

    return G, center


# @np_random_state(10)
def risk_elevation_layout(
        g,
        risks=None,
        alpha=2,
        k=None,
        pos=None,
        fixed=None,
        iterations=50,
        threshold=1e-4,
        weight="weight",
        scale=1,
        center=None,
        dim=2,
        seed=None,
        method=None,
        beta=1
):
    """Position nodes using Fruchterman-Reingold force-directed algorithm with
     elevation according to risks.

    The algorithm simulates a force-directed representation of the network
    treating edges as springs holding nodes close, while treating nodes
    as repelling objects, sometimes called an anti-gravity force in addition
    to an elevating force proportional to risks.
    Simulation continues until the positions are close to an equilibrium.

    There are some hard-coded values: minimal distance between
    nodes (0.01) and "temperature" of 0.1 to ensure nodes don't fly away.
    During the simulation, `k` helps determine the distance between nodes,
    though `scale` and `center` determine the size and place after
    rescaling occurs at the end of the simulation.

    Fixing some nodes doesn't allow them to move in the simulation.
    It also turns off the rescaling feature at the simulation's end.
    In addition, setting `scale` to `None` turns off rescaling.

    Parameters
    ----------
    g : NetworkX graph or list of nodes
        A position will be assigned to every node in G.

    risks : np 1d-array
        Estimated entity risks according to NRE

    alpha : float
        Coefficient for risk elevation force

    k : float (default=None)
        Optimal distance between nodes.  If None the distance is set to
        1/sqrt(n) where n is the number of nodes.  Increase this value
        to move nodes farther apart.

    pos : dict or None  optional (default=None)
        Initial positions for nodes as a dictionary with node as keys
        and values as a coordinate list or tuple.  If None, then use
        random initial positions.

    fixed : list or None  optional (default=None)
        Nodes to keep fixed at initial position.
        Nodes not in ``G.nodes`` are ignored.
        ValueError raised if `fixed` specified and `pos` not.

    iterations : int  optional (default=50)
        Maximum number of iterations taken

    threshold: float optional (default = 1e-4)
        Threshold for relative error in node position changes.
        The iteration stops if the error is below this threshold.

    weight : string or None   optional (default='weight')
        The edge attribute that holds the numerical value used for
        the edge weight.  Larger means a stronger attractive force.
        If None, then all edge weights are 1.

    scale : number or None (default: 1)
        Scale factor for positions. Not used unless `fixed is None`.
        If scale is None, no rescaling is performed.

    center : array-like or None
        Coordinate pair around which to center the layout.
        Not used unless `fixed is None`.

    dim : int
        Dimension of layout.

    seed : int, RandomState instance or None  optional (default=None)
        Set the random state for deterministic node layouts.
        If int, `seed` is the seed used by the random number generator,
        if numpy.random.RandomState instance, `seed` is the random
        number generator,
        if None, the random number generator is the RandomState instance used
        by numpy.random.

    method : str
        Force-directed algorithm to use. Default is fruchterman-reingold with risk term

        'davids': Neighbors attract each other with spring force, non-neighbors repel each other with columb force
         times shortest path distance

    beta: float
        Additional constant for some methods
        method = 'davids': Ratio of coulomb like force and spring force constants

    Returns
    -------
    pos : dict
        A dictionary of positions keyed by node

    """
    g, center = _process_params(g, center, dim)

    if fixed is not None:
        if pos is None:
            raise ValueError("nodes are fixed without positions given")
        for node in fixed:
            if node not in pos:
                raise ValueError("nodes are fixed without positions given")
        nfixed = {node: i for i, node in enumerate(g)}
        fixed = np.asarray([nfixed[node] for node in fixed if node in nfixed])

    if pos is not None:
        # Determine size of existing domain to adjust initial positions
        dom_size = max(coord for pos_tup in pos.values() for coord in pos_tup)
        if dom_size == 0:
            dom_size = 1
        pos_arr = seed.rand(len(g), dim) * dom_size + center

        for i, n in enumerate(g):
            if n in pos:
                pos_arr[i] = np.asarray(pos[n])
    else:
        pos_arr = None
        dom_size = 1

    if len(g) == 0:
        return {}
    if len(g) == 1:
        return {nx.utils.arbitrary_element(g.nodes()): center}

    try:
        # Sparse matrix
        if len(g) < 1000:  # sparse solver for large graphs
            raise ValueError
        mat_a = nx.to_scipy_sparse_array(g, weight=weight, dtype="f")
        if k is None and fixed is not None:
            # We must adjust k by domain size for layouts not near 1x1
            nnodes, _ = mat_a.shape
            k = dom_size / np.sqrt(nnodes)
        raise NotImplementedError("Modified Fruchterman Reingold not implemented for sparse matrices")

        # pos = _sparse_fruchterman_reingold(
        #     A, k, pos_arr, fixed, iterations, threshold, dim, seed
        # )

    except ValueError:
        mat_a = nx.to_numpy_array(g, weight=weight)
        if k is None and fixed is not None:
            # We must adjust k by domain size for layouts not near 1x1
            nnodes, _ = mat_a.shape
            k = dom_size / np.sqrt(nnodes)

        if method == 'davids':
            dists = list(nx.shortest_path_length(g))
            mat_d = np.zeros(mat_a.shape)
            nnodes = mat_a.shape[0]
            for i in range(nnodes):
                for j in range(nnodes):
                    assert dists[i][0] == list(g.nodes)[i], "Order of nodes due to nx.shortest_path_length is altered"
                    # print(dists)
                    mat_d[i, j] = dists[i][1][list(g.nodes)[j]]

            pos = _david_force_directed(
                mat_a, mat_d, risks=risks, alpha=alpha, k=k, pos=pos_arr, fixed=fixed, iterations=iterations,
                threshold=threshold,
                dim=dim, seed=seed, beta=beta)
        elif method == 'fuchterman-reingold' or method is None:  # None
            pos = _modified_fruchterman_reingold(
                mat_a, risks=risks, alpha=alpha, k=k, pos=pos_arr, fixed=fixed, iterations=iterations,
                threshold=threshold,
                dim=dim, seed=seed)
        else:
            raise NotImplementedError("Method {} not implemented".format(method))
    if fixed is None and scale is not None:
        pos = rescale_layout(pos, scale=scale) + center
    pos = dict(zip(g, pos))
    return pos


# @np_random_state(7)
def _modified_fruchterman_reingold(
        mat_a, risks=None, alpha=0.2, k=None, pos=None, fixed=None, iterations=50, threshold=1e-4, dim=2, seed=None
):
    # Position nodes in adjacency matrix A using Fruchterman-Reingold
    # Entry point for NetworkX graph is fruchterman_reingold_layout()
    np.random.seed(seed)

    if dim != 2:
        raise NotImplementedError("Only implemented for dim=2")

    try:
        nnodes, _ = mat_a.shape
    except AttributeError as err:
        msg = "fruchterman_reingold() takes an adjacency matrix as input"
        raise nx.NetworkXError(msg) from err

    if risks is None:
        risks = np.zeros(nnodes)
    else:
        risks -= np.mean(risks)
        risks /= np.linalg.norm(risks) if np.linalg.norm(risks) != 0 else np.zeros(nnodes)

    if pos is None:
        # random initial positions
        pos = np.asarray(np.random.rand(nnodes, dim), dtype=mat_a.dtype)
    else:
        # make sure positions are of same type as matrix
        pos = pos.astype(mat_a.dtype)

    # optimal distance between nodes
    if k is None:
        k = np.sqrt(1.0 / nnodes)
    # the initial "temperature"  is about .1 of domain area (=1x1)
    # this is the largest step allowed in the dynamics.
    # We need to calculate this in case our fixed positions force our domain
    # to be much bigger than 1x1
    t = max(max(pos.T[0]) - min(pos.T[0]), max(pos.T[1]) - min(pos.T[1])) * 0.1
    # simple cooling scheme.
    # linearly step down by dt on each iteration so last iteration is size dt.
    dt = t / (iterations + 1)
    delta = np.zeros((pos.shape[0], pos.shape[0], pos.shape[1]), dtype=mat_a.dtype)
    # the inscrutable (but fast) version
    # this is still O(V^2)
    # could use multilevel methods to speed this up significantly
    for iteration in range(iterations):
        # matrix of difference between points
        delta = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        # distance between points
        distance = np.linalg.norm(delta, axis=-1)
        # enforce minimum distance of 0.01
        np.clip(distance, 0.01, None, out=distance)
        # displacement "force"
        displacement = np.einsum(
            "ijk,ij->ik", delta, (k * k / distance ** 2 - mat_a * distance / k)  # Standard Fruchterman-Reingold
        )
        displacement[:, 1] += alpha * risks  # Modification due to risk elevation

        # update positions
        length = np.linalg.norm(displacement, axis=-1)
        length = np.where(length < 0.01, 0.1, length)  # Clip length
        delta_pos = np.einsum("ij,i->ij", displacement, t / length)  # Scale
        if fixed is not None:
            # don't change positions of fixed nodes
            delta_pos[fixed] = 0.0
        pos += delta_pos
        # cool temperature
        t -= dt
        if (np.linalg.norm(delta_pos) / nnodes) < threshold:
            break
    return pos


def _david_force_directed(
        mat_a, mat_d, risks=None, alpha=0.2, k=None, pos=None, fixed=None, iterations=50, threshold=1e-4, dim=2,
        seed=None,
        beta=1):
    # Position nodes in adjacency matrix A using a force directed algorithm

    np.random.seed(seed)

    if dim != 2:
        raise NotImplementedError("Only implemented for dim=2")

    try:
        nnodes, _ = mat_a.shape
    except AttributeError as err:
        msg = "fruchterman_reingold() takes an adjacency matrix as input"
        raise nx.NetworkXError(msg) from err

    if risks is None:
        risks = np.zeros(nnodes)
    else:
        risks -= np.mean(risks)
        risks /= np.linalg.norm(risks) if np.linalg.norm(risks) != 0 else np.zeros(nnodes)

    if pos is None:
        # random initial positions
        pos = np.asarray(np.random.rand(nnodes, dim), dtype=mat_a.dtype)
    else:
        # make sure positions are of same type as matrix
        pos = pos.astype(mat_a.dtype)

    # optimal distance between nodes
    if k is None:
        k = np.sqrt(1.0 / nnodes)
    # the initial "temperature"  is about .1 of domain area (=1x1)
    # this is the largest step allowed in the dynamics.
    # We need to calculate this in case our fixed positions force our domain
    # to be much bigger than 1x1
    t = max(max(pos.T[0]) - min(pos.T[0]), max(pos.T[1]) - min(pos.T[1])) * 0.1
    # simple cooling scheme.
    # linearly step down by dt on each iteration so last iteration is size dt.
    dt = t / (iterations + 1)
    delta = np.zeros((pos.shape[0], pos.shape[0], pos.shape[1]), dtype=mat_a.dtype)
    # the inscrutable (but fast) version
    # this is still O(V^2)
    # could use multilevel methods to speed this up significantly
    for iteration in range(iterations):
        # matrix of difference between points
        delta = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        # distance between points
        distance = np.linalg.norm(delta, axis=-1)
        # enforce minimum distance of 0.01
        np.clip(distance, 0.01, None, out=distance)
        # displacement "force"
        displacement = np.einsum(
            "ijk,ij->ik", delta, (beta * mat_d * (k / distance) - mat_a * distance / k)
            # "ijk,ij->ik", delta, (beta * mat_d * (k ** 2) / distance ** 2 - mat_a * distance / k)
        )  # TODO: Seems like its force times distance of each other node
        displacement[:, 1] += alpha * risks  # Modification due to risk elevation

        # update positions
        length = np.linalg.norm(displacement, axis=-1)
        length = np.where(length < 0.01, 0.1, length)  # Clip length
        delta_pos = np.einsum("ij,i->ij", displacement, t / length)  # Scale
        if fixed is not None:
            # don't change positions of fixed nodes
            delta_pos[fixed] = 0.0
        pos += delta_pos
        # cool temperature
        t -= dt
        if (np.linalg.norm(delta_pos) / nnodes) < threshold:
            break
    return pos


def get_layout_tree(g_connectivity, pos):
    """ Given the connectivity graph and layout positions of entities returns the visually pleasing tree that spans the
     network"""
    g_layout = nx.Graph()
    g_layout.add_nodes_from(g_connectivity)

    for u in g_layout:
        for v in g_layout:
            if u == v:
                continue
            g_layout.add_edge(u, v, weight=np.linalg.norm(pos[u] - pos[v]))
    return nx.minimum_spanning_tree(g_layout)
