import datetime
import matplotlib
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from matplotlib.lines import Line2D

from src.network_model import NetworkModel
from src.time_windowed import get_window


# Preprocessing Plotting Functions
def plot_flows(df, entities):
    """Plots the flows in the connection data df"""
    t_df = df.loc[
        df.loc[:, ' Source IP'].isin(entities) | df.loc[:, ' Destination IP'].isin(entities), [' Timestamp',
                                                                                               ' Source IP',
                                                                                               ' Destination IP']]
    t_df['Entity'] = (
            t_df.loc[:, ' Source IP'].isin([entities[0]]) | t_df.loc[:, ' Destination IP'].isin([entities[0]])).apply(
        lambda x: 'Entity 1' if x is True else 'Entity 2')
    t_df['y'] = t_df['Entity'].apply(lambda x: 0 if x == 'Entity 1' else 1)
    # t_df['y'] += nr.normal(0, 0.01, t_df.shape[0])
    t_df = t_df.iloc[:, :]

    # plt.figure(figsize= (12, 12))
    # sns.rugplot(data = t_df, x= ' Timestamp', hue = 'Entity', height=.1)
    sns.scatterplot(data=t_df, x=' Timestamp', y='y', hue='Entity')
    plt.xticks(rotation=45)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    plt.ylabel('')
    plt.title('Flows by Two Entities', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 0.65))


def plot_entity_signal(df, entities, conn_param='Num Packets Rec', time_window=10, time_scale='min',
                       date_col=' Timestamp'):
    """Plots the signals calculated for entities with respect to the connection parameter   """
    idx = df[' Source IP'].isin(entities) | df[' Destination IP'].isin(entities)
    t_df = df.loc[idx, :]
    idx1 = t_df[' Source IP'].isin([entities[0]]) | t_df[' Destination IP'].isin([entities[0]])
    idx_common = t_df[' Source IP'].isin(entities) | t_df[' Destination IP'].isin(entities)
    cpy_df = t_df.loc[idx_common, :]
    t_df = t_df.assign(Entity=idx1.apply(lambda x: 'Entity 1' if x is True else 'Entity 2'))
    cpy_df = cpy_df.assign(Entity='Entity 2')
    t_df = pd.concat((t_df, cpy_df), ignore_index=True)
    t_df = t_df.sort_values(by=[date_col])

    nm = NetworkModel()
    nm.read_flows(t_df, conn_param=conn_param, entity_names=list(entities), sync_window_size=time_window, time_scale=time_scale)
    vec1 = nm.samples[:, 0]
    vec2 = nm.samples[:, 1]
    epsy = -0.05 if conn_param == 'Activation' else 0
    vec2 += epsy

    times = []
    t_df = t_df.sort_values(by=[date_col])
    current_datetime = t_df.iloc[0][date_col]
    last_datetime = t_df.iloc[-1][date_col]
    while current_datetime <= last_datetime:
        window, current_datetime = get_window(current_datetime, t_df, time_window=time_window, time_scale=time_scale)
        times.append(current_datetime - datetime.timedelta(minutes=time_window / 2))

    sns.rugplot(data=t_df, x=' Timestamp', hue='Entity', height=.05)
    sns.lineplot(x=times, y=vec1, label='Entity 1', color='tab:blue')
    sns.lineplot(x=times, y=vec2, label='Entity 2', color='tab:orange')

    plt.xticks(rotation=45)
    plt.title('Number of Flows', fontsize=22, pad=20)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.ylabel('')
    plt.legend(loc='upper right', bbox_to_anchor=(1.4, 0.65))


def add_time_window_shades(ax, times, time_window, time_scale):
    """Given the axis adds the shaded regions illustrating windowing operation on raw flows"""
    half_win = datetime.timedelta(minutes=time_window/2) if time_scale == 'min' else \
        datetime.timedelta(seconds=time_window/2)
    y_min, y_max = ax.get_ylim()
    for i, t in enumerate(times):
        if i % 2 == 0:
            start = t - half_win
            stop = t + half_win
            ax.fill_between((start, stop), y_min, y_max, color=f'C{7}', alpha=0.2)


def plot_combined_flow_signals_old(df, entities, time_window=10, time_scale='min', date_col=' Timestamp',
                               conn_param1='Activation', conn_param2='Num Packets Received'):
    """Plots the figures depicting flows and two signals formed from them"""
    idx = df[' Source IP'].isin(entities) | df[' Destination IP'].isin(entities)
    t_df = df.loc[idx, :]
    idx0 = t_df[' Source IP'].isin([entities[0]]) | t_df[' Destination IP'].isin([entities[0]])
    idx_common = (t_df[' Source IP'].isin([entities[0]]) & t_df[' Destination IP'].isin([entities[1]]) ) | (t_df[' Source IP'].isin([entities[1]]) & t_df[' Destination IP'].isin([entities[0]]) )
    cpy_df = t_df.loc[idx_common, :]
    t_df = t_df.assign(Entity=idx0.apply(lambda x: 'Entity 1' if x is True else 'Entity 2'))
    cpy_df = cpy_df.assign(Entity='Entity 2')
    t_df = pd.concat((t_df, cpy_df), ignore_index=True)
    t_df = t_df.sort_values(by=[date_col])

    # 'Num Packets Rec'
    nm = NetworkModel()
    nm.read_flows(t_df, conn_param=conn_param1, entity_names=list(entities), sync_window_size=time_window,
                  time_scale=time_scale)
    vec_num1 = nm.samples[:, 0]
    vec_num2 = nm.samples[:, 1] - 0.05

    # Activation
    nm = NetworkModel()
    nm.read_flows(t_df, conn_param=conn_param2, entity_names=list(entities), sync_window_size=time_window,
                  time_scale=time_scale)
    vec_act1 = nm.samples[:, 0]
    vec_act2 = nm.samples[:, 1]

    times = []
    t_df = t_df.sort_values(by=[date_col])
    current_datetime = t_df.iloc[0][date_col]
    last_datetime = t_df.iloc[-1][date_col]
    while current_datetime <= last_datetime:
        window, current_datetime = get_window(current_datetime, t_df, time_window=time_window, time_scale=time_scale)
        times.append(current_datetime - datetime.timedelta(minutes=time_window / 2))

    # Plotting
    event_data = [t_df[t_df['Entity'] == 'Entity 1'].loc[:, date_col], t_df[t_df['Entity'] == 'Entity 2'].loc[:, date_col]]
    colors = [f'C{i}' for i in range(2)]
    fig, axs = plt.subplots(3, 1, gridspec_kw={'wspace': 0, 'hspace': 0})
    fig.suptitle('Flows by Entities')
    ax1, ax2, ax3 = axs
    ax1 = plt.subplot(311)
    # sns.scatterplot(data=t_df, x=' Timestamp', y='y', hue='Entity')
    # sns.stripplot(data=t_df, x=' Timestamp', y='y', hue='Entity', ax=ax1)
    ax1.eventplot(event_data, colors=colors, lineoffsets=[0, 2])
    add_time_window_shades(ax1, times, time_window, time_scale)
    # plt.xticks(rotation=45)


    # ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    # ax1.get_xaxis().set_ticks([])
    ax1.get_yaxis().set_ticks([])
    ax1.set_ylabel('')
    ax1.set_xlabel('Timestamps')
    # ax1.set_title('Flows by Entities')
    # ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 0.65))
    ax1.xaxis.tick_top()
    ax1.xaxis.set_label_position('top')

    ax2 = plt.subplot(312)
    # sns.rugplot(data=t_df, x=' Timestamp', hue='Entity', height=.05, ax=ax2)
    sns.lineplot(x=times, y=vec_num1, label='Entity 1', color='tab:blue', ax=ax2)
    sns.lineplot(x=times, y=vec_num2, label='Entity 2', color='tab:orange', ax=ax2)
    add_time_window_shades(ax2, times, time_window, time_scale)

    # ax2.set_title(conn_param1)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    #ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.get_xaxis().set_ticks([])
    ax2.set_ylabel(conn_param1)
    ax2.set_xlabel('')
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 0.65))

    ax3 = plt.subplot(313)
    # sns.rugplot(data=t_df, x=' Timestamp', hue='Entity', height=.05, ax=ax3)
    sns.lineplot(x=times, y=vec_act1, color='tab:blue', ax=ax3)
    sns.lineplot(x=times, y=vec_act2, color='tab:orange', ax=ax3)
    add_time_window_shades(ax3, times, time_window, time_scale)
    # ax3.get_legend().remove()

    # ax3.set_title(conn_param2)
    #ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    ax3.spines['left'].set_visible(False)
    ax3.get_xaxis().set_ticks([])
    ax3.set_ylabel(conn_param2)
    ax3.set_xlabel('')
    # ax3.legend()
    # ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 0.65))
    # plt.xticks(rotation=45)
    plt.tight_layout()


def plot_combined_flow_signals(df, entities, time_window=10, time_scale='min', date_col=' Timestamp',
                               conn_param1='Activation', conn_param2='Num Packets Received'):
    """Plots the figures depicting flows and two signals formed from them"""
    idx = df[' Source IP'].isin(entities) | df[' Destination IP'].isin(entities)
    t_df = df.loc[idx, :]
    idx0 = t_df[' Source IP'].isin([entities[0]]) | t_df[' Destination IP'].isin([entities[0]])
    idx_common = (t_df[' Source IP'].isin([entities[0]]) & t_df[' Destination IP'].isin([entities[1]]) ) | (t_df[' Source IP'].isin([entities[1]]) & t_df[' Destination IP'].isin([entities[0]]) )
    cpy_df = t_df.loc[idx_common, :]
    t_df = t_df.assign(Entity=idx0.apply(lambda x: 'Entity 1' if x is True else 'Entity 2'))
    cpy_df = cpy_df.assign(Entity='Entity 2')
    t_df = pd.concat((t_df, cpy_df), ignore_index=True)
    t_df = t_df.sort_values(by=[date_col])

    # Connection Parameter 1 (Activation)
    nm = NetworkModel()
    nm.read_flows(t_df, conn_param=conn_param1, entity_names=list(entities), sync_window_size=time_window,
                  time_scale=time_scale)
    vec_conn1_ent1 = nm.samples[:, 0]
    vec_conn1_ent2 = nm.samples[:, 1] - 0.05

    # Connection Parameter 2
    nm = NetworkModel()
    nm.read_flows(t_df, conn_param=conn_param2, entity_names=list(entities), sync_window_size=time_window,
                  time_scale=time_scale)
    vec_conn2_ent1 = nm.samples[:, 0]
    vec_conn2_ent2 = nm.samples[:, 1]

    times = []
    t_df = t_df.sort_values(by=[date_col])
    current_datetime = t_df.iloc[0][date_col]
    last_datetime = t_df.iloc[-1][date_col]
    while current_datetime <= last_datetime:
        window, current_datetime = get_window(current_datetime, t_df, time_window=time_window, time_scale=time_scale)
        times.append(current_datetime - datetime.timedelta(minutes=time_window / 2))

    # Plotting
    event_data = [t_df[t_df['Entity'] == 'Entity 1'].loc[:, date_col], t_df[t_df['Entity'] == 'Entity 2'].loc[:, date_col]]
    colors = [f'C{i}' for i in range(2)]
    fig = plt.figure(figsize=(9, 6))
    fig.suptitle('Flows by Entities', fontsize=18, weight='bold')
    ax = plt.gca()
    ax.eventplot(event_data, colors=colors, lineoffsets=[2, 3], linelengths=0.25)

    # Connection Parameter 1 (Activation)
    vec_conn1_ent1 = vec_conn1_ent1 * 2 - 1
    vec_conn1_ent2 = vec_conn1_ent2 * 2 - 1
    ax.plot(times, vec_conn1_ent1, label='Entity 1', color='C0')
    ax.plot(times, vec_conn1_ent2, label='Entity 2', color='C1')
    y_ticks1_loc = np.linspace(-1, 1, 2)
    y_ticks1_lbl = np.array([0, 1], dtype=int)

    # Connection Parameter 2
    all_signal = np.concatenate((vec_conn2_ent1, vec_conn2_ent2))
    scale = np.max(all_signal) - np.min(all_signal)
    all_min = np.min(all_signal)
    # Map to [0, 1]
    vec_conn2_ent1 = (vec_conn2_ent1 - all_min)/scale
    vec_conn2_ent2 = (vec_conn2_ent2 - all_min)/scale
    # Map to [-3.5, -1.5]
    vec_conn2_ent1 = vec_conn2_ent1 * 2 - 3.5
    vec_conn2_ent2 = vec_conn2_ent2 * 2 - 3.5

    # Plot
    ax.plot(times, vec_conn2_ent1, label='Entity 1', color='C0')
    ax.plot(times, vec_conn2_ent2, label='Entity 2', color='C1')
    y_ticks2_loc = np.array([-3, -2, -1]) - 0.5
    y_ticks2_lbl = np.linspace(all_min, all_min + scale, 3)
    shift = np.mod(scale/2, 100)
    y_ticks2_lbl -= np.linspace(0, 2*shift, 3)
    y_ticks2_lbl = y_ticks2_lbl.astype(int)

    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_yaxis().set_ticks(ticks=np.concatenate((y_ticks2_loc, y_ticks1_loc)),
                             labels=np.concatenate((y_ticks2_lbl, y_ticks1_lbl)))
    ax.set_ylabel('')
    ax.set_xlabel('Timestamps')
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    add_time_window_shades(ax, times, time_window, time_scale)

    # Annotations
    x_low, x_high = ax.get_xlim()
    x_scale = x_high - x_low
    ax.annotate('Flows', xy=(x_low - x_scale * 0.06, 2), annotation_clip=False, fontsize=12, rotation=90,
                style='italic')
    ax.annotate(conn_param1, xy=(x_low - x_scale * 0.065, -0.5), annotation_clip=False, fontsize=12, rotation=90,
                style='italic')
    ax.annotate(conn_param2, xy=(x_low - x_scale * 0.1, -4), annotation_clip=False, fontsize=12, rotation=90,
                style='italic')

    # Legend
    legend_elements = [Line2D([0], [0], color='C0', label='Entity 1'),
                       Line2D([0], [0], color='C1', label='Entity 2')]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 0.6))
    plt.tight_layout()


# Risk Estimates Over Time Plotting Functions
def risks_over_time_3d(mat_x_list, mat_p_list, t_graph=20, title='Sample Risk Estimates',
                       save_name='temp_plot.jpg'):
    """
    Plots the risk estimates overtime

    :param mat_x_list: List of mean of the estimates
    :param mat_p_list: List of covariance matrices of the risk estimates
    :param t_graph: Separation of time ticks in seconds
    :param title: Title of the plot
    :param save_name: If provided, saves the plot with this file name and extension
    :return fig: Figure handle of the plot
    """
    n_nodes = mat_x_list[-1].shape[0]

    shift, sep = 10, 1  # shift of x vals and separation btw time values
    x_min = np.min(np.array(mat_x_list))
    x_max = np.max(np.array(mat_x_list))
    p_min = np.min(np.array(mat_p_list))
    p_max = np.max(np.array(mat_p_list))

    fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1.5]})
    fig.set_figwidth(25)  # 25
    fig.set_figheight(11)  # 11

    axs[0].remove()
    ax_x = fig.add_subplot(1, 2, 1, projection='3d')
    ax_x.set_box_aspect([n_nodes, 1, n_nodes])
    axs[1].remove()
    ax_P = fig.add_subplot(1, 2, 2, projection='3d')
    ax_P.set_box_aspect([1.5, 1, 1])

    cmap_P = plt.cm.YlGnBu
    cmap_x = plt.cm.YlOrBr
    norm_P = matplotlib.colors.Normalize(vmin=np.abs(p_min), vmax=p_max)
    norm_x = matplotlib.colors.Normalize(vmin=np.abs(x_min), vmax=x_max)

    for k, x_kf, P_kf in zip(np.arange(len(mat_x_list)), mat_x_list, mat_p_list):
        xx_kf = np.hstack((x_kf, x_kf))
        Y_x, Z_x = np.meshgrid(np.arange(xx_kf.shape[1]) - shift, np.arange(xx_kf.shape[0])[::-1])
        X_x = np.ones(xx_kf.shape) * k * sep * t_graph

        cset_x = ax_x.plot_surface(X_x, Y_x, Z_x, rstride=1, cstride=1, facecolors=cmap_x(norm_x(x_kf)),
                                   cmap=cmap_x, shade=False)

    ax_x.grid(False)
    ax_x.view_init(elev=20, azim=-55)
    cbar_x = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm_x, cmap=cmap_x), ax=ax_x, shrink=0.5)  # , pad=0.2)
    cbar_x.set_label('Mean of Risk Estimates (' + r'$\hat{\mathbf{x}}_{t|t}$)', fontsize=24)
    cbar_x.ax.tick_params(labelsize=18)
    cbar_x.ax.yaxis.offsetText.set(size=18)

    ax_x.set_yticks([-shift])
    ax_x.set_yticklabels([''])
    ax_x.set_title(r'$\hat{\mathbf{x}}_{t|t}$', fontsize=26, x=0.65)

    plt.xticks(fontsize=14)
    ax_x.set_zticks(np.arange(n_nodes, step=2))
    ax_x.set_zticklabels(np.arange(n_nodes, step=2)[::-1])
    ax_x.tick_params(axis='x', labelsize=18, pad=-3)
    ax_x.tick_params(axis='z', labelsize=18)
    ax_x.set_xlabel('t (s)', fontsize=24, labelpad=20)
    ax_x.set_zlabel('Entity Index', fontsize=24, labelpad=20)
    # ax_x.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax_x.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax_x.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # P_kf
    for k, x_kf, P_kf in zip(np.arange(len(mat_x_list)), mat_x_list, mat_p_list):
        Y_P, Z_P = np.meshgrid(np.arange(P_kf.shape[0]), np.arange(P_kf.shape[1])[::-1])
        X_P = np.ones(P_kf.shape) * k * sep * t_graph

        cset_P = ax_P.plot_surface(X_P, Y_P, Z_P, rstride=1, cstride=1, facecolors=cmap_P(norm_P(P_kf)),
                                   cmap=cmap_P, shade=False, norm=norm_P)

    ax_P.grid(False)
    ax_P.view_init(elev=20, azim=-60)
    cbar_P = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm_P, cmap=cmap_P), ax=ax_P, shrink=0.5, pad=0.1)
    cbar_P.set_label('Covariance Matrix (' + r'$\mathbf{P}_{t|t}$)', fontsize=24)
    cbar_P.ax.tick_params(labelsize=18)
    cbar_P.ax.yaxis.offsetText.set(size=18)

    ax_P.set_title(r'$\mathbf{P}_{t|t}$', fontsize=26, x=0.65)

    plt.xticks(fontsize=14)
    ax_P.set_zticks(np.arange(x_kf.shape[0], step=2))
    ax_P.set_zticklabels(np.arange(x_kf.shape[0], step=2)[::-1])
    ax_P.set_yticks(np.arange(x_kf.shape[0], step=2))
    ax_P.set_yticklabels(np.arange(x_kf.shape[0], step=2)[::1])
    ax_P.tick_params(axis='x', labelsize=18, pad=-3)
    ax_P.tick_params(axis='z', labelsize=18)
    ax_P.tick_params(axis='y', labelsize=16, rotation=-45)
    ax_P.set_xlabel('t (s)', fontsize=24, labelpad=10)
    ax_P.set_zlabel('Entity Index', fontsize=24, labelpad=15)
    ax_P.set_ylabel('Entity Index', fontsize=24, labelpad=15)

    plt.tight_layout()
    plt.suptitle(title, fontsize=30)
    if save_name is not None:
        plt.savefig(save_name)  # , transparent=True)
    plt.show()
    return fig


# Safe Routing Plotting Functions
def polar2cartesian(r, theta, units='deg'):
    """Return the cartesian coordinates corresponding to the polar coordinates"""
    if units == 'deg':
        theta = theta * np.pi / 180
    else:
        assert units == 'rad', "Implemented for units 'deg' or 'rad'"
    return r * np.cos(theta), r * np.sin(theta)


def cartesian_dist_plot(g, distances, risks, title='', dx=1, dy=1, n_points=1000, plot=True, show_names=True):
    """
    Depicts the network according to the calculated distances to route from source to every other entity. Fits the
    entities to integer cartesian coordinates where the source is left most and every other entity is distributed according
    to their distance.

    :param g: NetworkX graph which stands for the communication graph of the network
    :param distances: Dictionary of distances or risk-levels from the source. The Dictionary is ordered with increasing
        distances
    :param risks: Dictionary of entity risks
    :param title: Title of the plot
    :param dx: Increment in x direction
    :param dy: Increment in y direction
    :param plot: If True, show the plot
    :param n_points: Number of points to draw vertical lines
    :param show_names: If True, shows the names of entities as well
    :return fig: Figure of the network plot
    """
    source = list(distances.keys())[0]
    vals, indices, counts = np.unique(list(distances.values()), return_inverse=True, return_counts=True)

    x_cur, y_cur = 0, 0
    idx_prev = -1
    fig, ax = plt.subplots(figsize=(9, 6))
    pos = {}
    max_num = 0
    x_curs = []
    for i, node in enumerate(distances.keys()):
        idx = indices[i]
        num = counts[indices[i]]  # Number of same distance entities
        max_num = num if num > max_num else max_num

        # Update coordinates
        if idx != idx_prev:
            y_init = - (num-1)/2 * dy
            x_cur, y_cur = x_cur + dx, y_init
            x_curs.append(x_cur)
        else:
            x_cur, y_cur = x_cur, y_cur + dy
        pos[node] = np.array([x_cur, y_cur])
        idx_prev = idx

    # Plot lines
    ys = np.linspace(-(max_num/2 + 0.5), max_num/2 + 0.5, n_points) * dy
    for x_cur in x_curs:
        xs = np.array([x_cur for _ in ys])
        ax.plot(xs, ys, 'tab:gray', alpha=.2)

    # Plot the rest
    cmap = plt.cm.Reds  # plt.cm.Reds
    node_clr = [risks[node] for node in g.nodes]
    norm = matplotlib.colors.Normalize(vmin=min(node_clr), vmax=max(node_clr))

    if show_names:
        pos_label = {node:pos[node] + (0, 0.45) for node in pos}
        nx.draw_networkx_labels(g, pos_label, ax=ax, clip_on=False, font_size=10)
    most_nodes = nx.draw_networkx_nodes(g, pos, cmap=cmap, node_color=node_clr, ax=ax)
    most_nodes.set_edgecolor('black')
    most_nodes.set_linewidth(2)
    nx.draw_networkx_edges(g, pos, width=1.6, edge_color='darkred', ax=ax)

    # Draw the source
    options = {"node_size": 350, "node_shape": 's', "node_color": cmap(norm(risks[source]))}
    src_node = nx.draw_networkx_nodes(g, pos, nodelist=[source], ax=ax, **options)
    src_node.set_edgecolor('black')
    src_node.set_linewidth(2)

    if not show_names:
        ax.annotate('Source', xy=pos[source] + (-.28, .45), size=9, annotation_clip=False)
    cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.05, shrink=0.6)
    cbar.set_label('Entity Risks', labelpad=-55)
    ax.set_title(title)
    plt.box(False)

    # Set x ticks
    ax.spines['top'].set_visible(True)
    ax.set_xlabel('Increasing Risk Levels')
    ax.get_xaxis().set_ticks(ticks=x_curs)

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
