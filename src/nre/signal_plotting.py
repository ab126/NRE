import datetime

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from .network_connectivity import ConnectivityUnit
from .time_windowed import get_window


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

    nm = ConnectivityUnit()
    nm.read_flows(t_df, conn_param=conn_param, entity_names=list(entities), sync_window_size=time_window,
                  time_scale=time_scale)
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
    half_win = datetime.timedelta(minutes=time_window / 2) if time_scale == 'min' else \
        datetime.timedelta(seconds=time_window / 2)
    y_min, y_max = ax.get_ylim()
    for i, t in enumerate(times):
        if i % 2 == 0:
            start = t - half_win
            stop = t + half_win
            ax.fill_between((start, stop), y_min, y_max, color=f'C{7}', alpha=0.2)


def plot_combined_flow_signals_old(df, entities, time_window=10, time_scale='min', date_col=' Timestamp',
                                   conn_param1='Activation', conn_param2='NPR', str1=None, str2=None):
    """Plots the figures depicting flows and two signals formed from them"""
    if str1 is None:
        str1 = conn_param1
    if str2 is None:
        str2 = conn_param2
    idx = df[' Source IP'].isin(entities) | df[' Destination IP'].isin(entities)
    t_df = df.loc[idx, :]
    idx0 = t_df[' Source IP'].isin([entities[0]]) | t_df[' Destination IP'].isin([entities[0]])
    idx_common = (t_df[' Source IP'].isin([entities[0]]) & t_df[' Destination IP'].isin([entities[1]])) | (
            t_df[' Source IP'].isin([entities[1]]) & t_df[' Destination IP'].isin([entities[0]]))
    cpy_df = t_df.loc[idx_common, :]
    t_df = t_df.assign(Entity=idx0.apply(lambda x: 'Entity 1' if x is True else 'Entity 2'))
    cpy_df = cpy_df.assign(Entity='Entity 2')
    t_df = pd.concat((t_df, cpy_df), ignore_index=True)
    t_df = t_df.sort_values(by=[date_col])

    # Connection Parameter 1 (Activation)
    cu = ConnectivityUnit()
    cu.read_flows(t_df, conn_param=conn_param1, entity_names=list(entities), sync_window_size=time_window,
                  time_scale=time_scale)
    vec_conn1_ent1 = cu.samples[:, 0]
    vec_conn1_ent2 = cu.samples[:, 1] - 0.05

    # Connection Parameter 2
    cu = ConnectivityUnit()
    cu.read_flows(t_df, conn_param=conn_param2, entity_names=list(entities), sync_window_size=time_window,
                  time_scale=time_scale)
    vec_conn2_ent1 = cu.samples[:, 0]
    vec_conn2_ent2 = cu.samples[:, 1]

    times = []
    t_df = t_df.sort_values(by=[date_col])
    current_datetime = t_df.iloc[0][date_col]
    last_datetime = t_df.iloc[-1][date_col]
    while current_datetime <= last_datetime:
        window, current_datetime = get_window(current_datetime, t_df, time_window=time_window, time_scale=time_scale)
        times.append(current_datetime - datetime.timedelta(minutes=time_window / 2))

    # Plotting flows yaxis in [3, 4]
    event_data = [t_df[t_df['Entity'] == 'Entity 1'].loc[:, date_col],
                  t_df[t_df['Entity'] == 'Entity 2'].loc[:, date_col]]
    colors = [f'C{i}' for i in range(2)]
    fig = plt.figure(figsize=(9, 6))
    fig.suptitle('Flows by Entities', fontsize=18, weight='bold')
    ax = plt.gca()
    ax.eventplot(event_data, colors=colors, lineoffsets=[3, 4], linelengths=0.25)

    # Connection Parameter 1 (Activation) yaxis in [-1, 1]
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
    vec_conn2_ent1 = (vec_conn2_ent1 - all_min) / scale
    vec_conn2_ent2 = (vec_conn2_ent2 - all_min) / scale
    # Map to [-4.5, -2.5]
    vec_conn2_ent1 = vec_conn2_ent1 * 2 - 4.5 # 4.5
    vec_conn2_ent2 = vec_conn2_ent2 * 2 - 4.5

    # Plot
    ax.plot(times, vec_conn2_ent1, label='Entity 1', color='C0')
    ax.plot(times, vec_conn2_ent2, label='Entity 2', color='C1')
    y_ticks2_loc = np.array([0, 1, 2]) - 4.5
    y_ticks2_lbl = np.linspace(all_min, all_min + scale, 3)
    shift = np.mod(scale / 2, 100)
    y_ticks2_lbl -= np.linspace(0, 2 * shift, 3)
    y_ticks2_lbl = y_ticks2_lbl.astype(int)

    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_yaxis().set_ticks(ticks=np.concatenate((y_ticks2_loc, y_ticks1_loc)),
                             labels=np.concatenate((y_ticks2_lbl, y_ticks1_lbl)))
    ax.set_ylabel('')
    ax.set_xlabel('Timestamps $\it{(mm/dd/hh)}$')
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    add_time_window_shades(ax, times, time_window, time_scale)

    # Annotations
    x_low, x_high = ax.get_xlim()
    x_scale = x_high - x_low
    ax.annotate('Flows', xy=(x_low - x_scale * 0.02, 3.3), annotation_clip=False, fontsize=12, rotation=90,
                style='italic')
    ax.annotate(str1, xy=(x_low - x_scale * 0.065, -0.8), annotation_clip=False, fontsize=12, rotation=90,
                style='italic')
    ax.annotate(str2, xy=(x_low - x_scale * 0.13, -5), annotation_clip=False, fontsize=12, rotation=90,
                style='italic')

    # Plot separating line
    x_start, x_end = times[0] - pd.DateOffset(hours=0.7), times[-1] + pd.DateOffset(hours=0.6)
    plt.xlim(ax.get_xlim())
    plt.ylim(ax.get_ylim())
    plt.plot([x_start, x_end], [2, 2], 'k', linewidth=0.6, clip_on=False)
    plt.plot([x_start, x_end], [-2, -2], 'k', linewidth=0.6, clip_on=False)

    # Legend
    legend_elements = [Line2D([0], [0], color='C0', label='Entity 1'),
                       Line2D([0], [0], color='C1', label='Entity 2')]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 0.6))
    plt.tight_layout()


def plot_combined_flow_signals(df, entities, time_window=10, time_scale='min', date_col=' Timestamp',
                               conn_params=None, conn_strings=None, figsize=(9, 6)):
    """Plots the figures depicting flows and two signals formed from them"""
    if conn_params is None:
        conn_params = []
    if conn_strings is None:
        conn_strings = conn_params

    idx = df[' Source IP'].isin(entities) | df[' Destination IP'].isin(entities)
    t_df = df.loc[idx, :]
    idx0 = t_df[' Source IP'].isin([entities[0]]) | t_df[' Destination IP'].isin([entities[0]])
    idx_common = (t_df[' Source IP'].isin([entities[0]]) & t_df[' Destination IP'].isin([entities[1]])) | (
            t_df[' Source IP'].isin([entities[1]]) & t_df[' Destination IP'].isin([entities[0]]))
    cpy_df = t_df.loc[idx_common, :]
    t_df = t_df.assign(Entity=idx0.apply(lambda x: 'Entity 1' if x is True else 'Entity 2'))
    cpy_df = cpy_df.assign(Entity='Entity 2')
    t_df = pd.concat((t_df, cpy_df), ignore_index=True)
    t_df = t_df.sort_values(by=[date_col])

    times = []
    t_df = t_df.sort_values(by=[date_col])
    current_datetime = t_df.iloc[0][date_col]
    last_datetime = t_df.iloc[-1][date_col]
    while current_datetime <= last_datetime:
        window, current_datetime = get_window(current_datetime, t_df, time_window=time_window, time_scale=time_scale)
        times.append(current_datetime - datetime.timedelta(minutes=time_window / 2))

    # Plotting flows yaxis in [0, 1] # [3, 4]
    event_data = [t_df[t_df['Entity'] == 'Entity 1'].loc[:, date_col],
                  t_df[t_df['Entity'] == 'Entity 2'].loc[:, date_col]]
    colors = [f'C{i}' for i in range(2)]
    fig = plt.figure(figsize=figsize)
    fig.suptitle('Flows by Endpoints', fontsize=18, weight='bold')
    ax = plt.gca()
    ax.eventplot(event_data, colors=colors, lineoffsets=[0, 1], linelengths=0.25)

    # Plot separating Lines
    x_start, x_end = times[0] - pd.DateOffset(hours=0.7), times[-1] + pd.DateOffset(hours=0.6)

    y_ticks_locs = np.empty((0,))
    y_ticks_lbls = np.empty((0,))
    signal_lower_bound = 0
    signal_size = 2
    for i, conn_param in enumerate(conn_params):

        # Connection Parameter 1 (Activation)
        cu = ConnectivityUnit()
        cu.read_flows(t_df, conn_param=conn_param, entity_names=list(entities), sync_window_size=time_window,
                      time_scale=time_scale)
        vec_conn_ent1 = cu.samples[:, 0]
        vec_conn_ent2 = cu.samples[:, 1]
        if conn_param == 'Activation':
            vec_conn_ent2 -= 0.05  # Shift overlapping curves

        # Compute location on plot
        signal_upper_bound = -1.5 - 3*i
        signal_lower_bound = signal_upper_bound - signal_size

        # Connection Parameter 1 (Activation) yaxis in [-1, 1]
        if conn_param == 'Activation':
            # Map to [low, high]
            vec_conn_ent1 = vec_conn_ent1 * signal_size + signal_lower_bound
            vec_conn_ent2 = vec_conn_ent2 * signal_size + signal_lower_bound

            # Plot
            ax.plot(times, vec_conn_ent1, label='Entity 1', color='C0')
            ax.plot(times, vec_conn_ent2, label='Entity 2', color='C1')
            y_ticks_loc = np.array([signal_lower_bound, signal_upper_bound], dtype=float)
            y_ticks_lbl = np.array([0, 1], dtype=int)

        # Connection Parameter 2
        else:
            all_signal = np.concatenate((vec_conn_ent1, vec_conn_ent2))
            scale_signal = np.max(all_signal) - np.min(all_signal)
            all_min = np.min(all_signal)
            # Map to [0, 1]
            vec_conn_ent1 = (vec_conn_ent1 - all_min) / scale_signal
            vec_conn_ent2 = (vec_conn_ent2 - all_min) / scale_signal
            # Map to [low, high]
            vec_conn_ent1 = vec_conn_ent1 * signal_size + signal_lower_bound
            vec_conn_ent2 = vec_conn_ent2 * signal_size + signal_lower_bound

            # Plot
            ax.plot(times, vec_conn_ent1, label='Entity 1', color='C0')
            ax.plot(times, vec_conn_ent2, label='Entity 2', color='C1')
            y_ticks_loc = np.array([0, signal_size/2, signal_size]) + signal_lower_bound
            y_ticks_lbl = np.linspace(all_min, all_min + scale_signal, 3)
            shift = np.mod(scale_signal / 2, 100)
            y_ticks_lbl -= np.linspace(0, 2 * shift, 3)
            y_ticks_lbl = y_ticks_lbl.astype(int)
        y_ticks_locs = np.concatenate((y_ticks_locs, y_ticks_loc))
        y_ticks_lbls = np.concatenate((y_ticks_lbls, y_ticks_lbl))

        # Plot separating lines
        plt.plot([x_start, x_end], [signal_upper_bound + signal_size/4, signal_upper_bound + signal_size/4], 'k',
                 linewidth=0.6, clip_on=False)

    # Adjustments
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_yaxis().set_ticks(ticks=y_ticks_locs, labels=y_ticks_lbls)
    #ax.get_yaxis().set_tick_params(pad=-20)
    ax.set_ylabel('')
    ax.set_xlabel('Timestamps $\it{(mm/dd/hh)}$')
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    add_time_window_shades(ax, times, time_window, time_scale)

    # Annotations
    x_low, x_high = ax.get_xlim()
    x_scale = x_high - x_low
    ax.annotate('Flows', xy=(x_low - x_scale * 0.02, 0.3), annotation_clip=False, fontsize=12, rotation=90,
                style='italic')

    for i, conn_param in enumerate(conn_params):
        str_conn = conn_strings[i]
        signal_upper_bound = -1.5 - 3 * i
        signal_lower_bound = signal_upper_bound - signal_size
        ax.annotate(str_conn, xy=(x_low - x_scale * 0.21, signal_lower_bound), annotation_clip=False, fontsize=12,
                    rotation=90, style='italic')  # 0.13

    # Legend
    legend_elements = [Line2D([0], [0], color='C0', label='Endpoint 1'),
                       Line2D([0], [0], color='C1', label='Endpoint 2')]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 0.6)) # 1.15
    plt.tight_layout()
    return ax


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
    k_steps = len(mat_x_list)

    fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1.5]})
    fig.set_figwidth(25)  # 25
    fig.set_figheight(11)  # 11

    axs[0].remove()
    ax_x = fig.add_subplot(1, 2, 1, projection='3d')
    ax_x.set_box_aspect([n_nodes, 1, n_nodes])
    axs[1].remove()
    ax_p = fig.add_subplot(1, 2, 2, projection='3d')
    ax_p.set_box_aspect([1.5, 1, 1])

    shift, sep = 10, 1  # shift of x vals and separation btw time values
    x_min = np.min(np.array(mat_x_list))
    x_max = np.max(np.array(mat_x_list))
    p_min = np.min(np.array(mat_p_list))
    p_max = np.max(np.array(mat_p_list))

    cmap_x = plt.cm.YlOrBr
    cmap_p = plt.cm.YlGnBu
    norm_x = matplotlib.colors.Normalize(vmin=np.abs(x_min), vmax=x_max)
    norm_p = matplotlib.colors.Normalize(vmin=np.abs(p_min), vmax=p_max)

    for k, x_kf, p_kf in zip(np.arange(k_steps), mat_x_list, mat_p_list):
        y_x, z_x = np.meshgrid(np.arange(2) - shift, np.arange(n_nodes)[::-1])
        x_x = np.ones((n_nodes, 2)) * k * sep * t_graph

        cset_x = ax_x.plot_surface(x_x, y_x, z_x, rstride=1, cstride=1, facecolors=cmap_x(norm_x(x_kf)),
                                   cmap=cmap_x, shade=False)

    ax_x.grid(False)
    ax_x.view_init(elev=20, azim=-55)
    cbar_x = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm_x, cmap=cmap_x), ax=ax_x, shrink=0.5)  # , pad=0.2)
    cbar_x.set_label('Mean of Risk Estimates (' + r'$\hat{\mathbf{x}}_{t|t}$)', fontsize=24, rotation=-90, labelpad=30)
    cbar_x.ax.tick_params(labelsize=18)
    cbar_x.ax.yaxis.offsetText.set(size=18)

    ax_x.set_yticks([-shift])
    ax_x.set_yticklabels([''])
    ax_x.set_title(r'$\hat{\mathbf{x}}_{t|t}$', fontsize=26, x=0.65)

    plt.xticks(fontsize=14)
    ax_x.set_zticks(np.arange(n_nodes, step=2))
    ax_x.set_zticklabels(np.arange(n_nodes, step=2)[::-1])
    ax_x.tick_params(axis='x', labelsize=18, pad=-3)
    ax_x.set_xticks([i * t_graph for i in range(k_steps)])
    ax_x.tick_params(axis='z', labelsize=18)
    ax_x.set_xlabel('t (s)', fontsize=24, labelpad=20)
    ax_x.zaxis.set_rotate_label(False)
    ax_x.set_zlabel('Entity Index', fontsize=24, labelpad=20, rotation=-90)

    # ax_x.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax_x.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax_x.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # P_kf
    for k, x_kf, p_kf in zip(np.arange(k_steps), mat_x_list, mat_p_list):
        y_p, z_p = np.meshgrid(np.arange(n_nodes), np.arange(n_nodes)[::-1])
        x_p = np.ones((n_nodes, n_nodes)) * k * sep * t_graph

        cset_p = ax_p.plot_surface(x_p, y_p, z_p, rstride=1, cstride=1, facecolors=cmap_p(norm_p(p_kf)),
                                   cmap=cmap_p, shade=False, norm=norm_p)

    ax_p.grid(False)
    ax_p.view_init(elev=20, azim=-60)
    cbar_p = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm_p, cmap=cmap_p), ax=ax_p, shrink=0.5, pad=0.1)
    cbar_p.set_label('Covariance Matrix (' + r'$\mathbf{P}_{t|t}$)', fontsize=24, rotation=-90, labelpad=30)
    cbar_p.ax.tick_params(labelsize=18)
    cbar_p.ax.yaxis.offsetText.set(size=18)

    ax_p.set_title(r'$\mathbf{P}_{t|t}$', fontsize=26, x=0.65)

    plt.xticks(fontsize=14)
    ax_p.set_zticks(np.arange(x_kf.shape[0], step=2))
    ax_p.set_zticklabels(np.arange(x_kf.shape[0], step=2)[::-1])
    ax_p.set_yticks(np.arange(x_kf.shape[0], step=2))
    ax_p.set_yticklabels(np.arange(x_kf.shape[0], step=2)[::1])
    ax_p.tick_params(axis='x', labelsize=18, pad=-3)
    ax_p.set_xticks([i * t_graph for i in range(k_steps)])
    ax_p.tick_params(axis='z', labelsize=18)
    ax_p.tick_params(axis='y', labelsize=16, rotation=-45)
    ax_p.set_xlabel('t (s)', fontsize=24, labelpad=10)
    ax_p.zaxis.set_rotate_label(False)
    ax_p.set_zlabel('Entity Index', fontsize=24, labelpad=15, rotation=-90)
    ax_p.set_ylabel('Entity Index', fontsize=24, labelpad=15)

    # plt.tight_layout()
    plt.suptitle(title, fontsize=30)
    if save_name is not None:
        plt.savefig(save_name)  # , transparent=True)
    plt.show()
    return fig


def risks_over_time_2d(mat_x_list, mat_p_list, mat_f, t_graph=20, title='', save_name='temp_plot.jpg',
                       show_border=False):
    """
    Plots the risk estimates overtime, 2d layout

    :param mat_x_list: List of mean of the estimates
    :param mat_p_list: List of covariance matrices of the risk estimates
    :param mat_f: Connectivity Graph Matrix F
    :param t_graph: Separation of time ticks in seconds
    :param title: Title of the plot
    :param save_name: If provided, saves the plot with this file name and extension
    :param show_border: Show some borders for fixing layout
    :return fig: Figure handle of the plot
    """
    n_nodes = mat_x_list[-1].shape[0]
    k_steps = len(mat_x_list)
    assert k_steps > 0

    fig = plt.figure(figsize=(16, 6))
    spec = fig.add_gridspec(3, k_steps + 3, height_ratios=[.4, 1.8, 2.4],
                            width_ratios=[1.5, .75] + [1 for _ in range(k_steps)] + [.5], hspace=0.2, wspace=0.4)

    x_min = np.min(np.array(mat_x_list))
    x_max = np.max(np.array(mat_x_list))
    p_min = np.min(np.array(mat_p_list))
    p_max = np.max(np.array(mat_p_list))

    cmap_x = plt.cm.YlOrBr  # YlOrBr
    cmap_p = plt.cm.YlGnBu  # YlGnBu
    cmap_f = 'Blues'
    norm_x = matplotlib.colors.Normalize(vmin=np.abs(x_min), vmax=x_max)
    norm_p = matplotlib.colors.Normalize(vmin=np.abs(p_min), vmax=p_max)
    norm_f = matplotlib.colors.Normalize(vmin=np.min(mat_f), vmax=np.max(mat_f))

    # Connectivity Graph
    ax_f = fig.add_subplot(spec[:, 0])
    ax_f.matshow(mat_f, cmap='Blues')
    cbar_f = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm_f, cmap=cmap_f), ax=ax_f, shrink=0.4)
    cbar_f.ax.tick_params(labelsize=16)
    ax_f.set_title(r'$\mathbf{F}^{(t)}$', fontsize=20)
    ax_f.set_xlabel('Entity Index', weight='bold', fontsize=16)
    ax_f.set_ylabel('Entity Index', weight='bold', fontsize=16)
    ax_f = plt.gca()
    ax_f.xaxis.set_ticks_position('bottom')

    # t axis
    ax_t = fig.add_subplot(spec[0, 1:])
    ax_t.plot(1, 0, ls="", marker=">", ms=10, color="k", clip_on=False)
    ax_t.plot([0, 1], [0, 0], 'k')
    # Ticks
    xts = [.14, .365, .595, .82]  # , .86]
    shift = 0.08  # How much to shift the start of the time line
    for k, xt in enumerate(xts):
        ax_t.plot([xt, xt], [-1, 1], 'k')
        ax_t.text(xt, -2, str(k * t_graph), clip_on=False, fontsize=18, ha="center", va="top")
    ax_t.text(1.01, 0, "t (s)", ha="left", va="center", fontsize=20, clip_on=False)
    x_low, x_high = ax_t.get_xlim()
    ax_t.set_xlim([x_low - shift, x_high])
    ax_t.set_ylim([-5, 5])
    if not show_border:
        ax_t.axis("off")

    # x_k
    x_axs = []
    ax_x_0 = fig.add_subplot(spec[1, 1])
    x_axs.append(ax_x_0)
    if not show_border:
        ax_x_0.axis('off')
    ax_x_0.text(0.5, 0.5, r'$\hat{\mathbf{x}}_{t|t}$:', fontsize=26, ha="center", va="center")
    ax_x_0.set_xlim([0, 0.7])

    for k, x_kf, p_kf in zip(np.arange(k_steps) + 1, mat_x_list, mat_p_list):
        ax_x_k = fig.add_subplot(spec[1, k + 1])
        x_axs.append(ax_x_k)
        ax_x_k.matshow(x_kf, vmin=x_min, vmax=x_max, cmap=cmap_x)

        if k == k_steps:
            ax_x_k.set_ylabel('Entity Index', fontsize=18, labelpad=20, rotation=-90)

        ax_x_k.tick_params(axis='x', bottom=False, top=False, labeltop=False)
        ax_x_k.set_yticks(np.arange(n_nodes, step=4))
        ax_x_k.tick_params(axis='y', labelsize=18)
        ax_x_k.yaxis.tick_right()
        ax_x_k.yaxis.set_label_position("right")

    ax_x_last = fig.add_subplot(spec[1, -1])
    ax_x_last.axis('off')
    cbar_x = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm_x, cmap=cmap_x), ax=ax_x_last,  # location='left',
                          shrink=1, pad=-1.5)
    cbar_x.set_label('Mean of Risk\nEstimates ' + r'($\hat{\mathbf{x}}_{t|t}$)', fontsize=20, rotation=-90,
                     labelpad=55)  # -40
    cbar_x.ax.yaxis.set_ticks_position('left')
    cbar_x.ax.tick_params(labelsize=18)
    cbar_x.ax.yaxis.offsetText.set(size=18)

    # P_kf
    ax_p_0 = fig.add_subplot(spec[2, 1])
    if not show_border:
        ax_p_0.axis('off')
    ax_p_0.text(0.5, 0.5, r'$\mathbf{P}_{t|t}$:', fontsize=26, ha="center", va="center")
    ax_p_0.set_xlim([0, 0.7])

    for k, x_kf, p_kf in zip(np.arange(k_steps) + 1, mat_x_list, mat_p_list):
        ax_p_k = fig.add_subplot(spec[2, k + 1])
        ax_p_k.matshow(p_kf, vmin=p_min, vmax=p_max, cmap=cmap_p)

        if k == k_steps:
            ax_p_k.set_ylabel('Entity Index', fontsize=18, labelpad=20, rotation=-90)

        ax_p_k.xaxis.set_label_position("bottom")
        ax_p_k.set_xticks(np.arange(n_nodes, step=4))
        ax_p_k.tick_params(axis='x', labelsize=18)
        ax_p_k.xaxis.tick_bottom()
        ax_p_k.set_yticks(np.arange(n_nodes, step=4))
        ax_p_k.tick_params(axis='y', labelsize=18)
        ax_p_k.yaxis.tick_right()
        ax_p_k.yaxis.set_label_position("right")

    ax_p_last = fig.add_subplot(spec[2, -1])
    ax_p_last.axis('off')
    cbar_p = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm_p, cmap=cmap_p), ax=ax_p_last, shrink=1)  # , pad=-10)
    cbar_p.set_label('Covariance\nMatrix ' + r'($\mathbf{P}_{t|t}$)', fontsize=20, rotation=-90, labelpad=50)
    cbar_p.ax.yaxis.set_ticks_position('left')
    cbar_p.ax.tick_params(labelsize=18)
    cbar_p.ax.yaxis.offsetText.set(size=18)

    # plt.suptitle(title, fontsize=30)
    if save_name is not None:
        plt.savefig(save_name)  # , transparent=True)
    plt.show()
    return fig
