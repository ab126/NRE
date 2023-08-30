import pandas as pd
import time
import datetime
#from .custom_graph import *
import numpy as np

from src.custom_graph import get_Graph, get_data


# TODO: Probably lot of abundance here. fix Imports as well

def time_func(f, args=[], kw_args={}):
    """
    Times one call to f with args, kw_args.

    Arguments:
    f       -- the function to be timed
    args    -- list of arguments to pass to f
    kw_args -- dictionary of keyword arguments to pass to f.

    Returns:
    a tuple containing the result of the call and the time it
    took (in seconds).

    """
    start_time = time.time()
    result = f(*args, **kw_args)
    end_time = time.time()

    return result, end_time - start_time


def get_window(start_datetime, df, time_window=2, date_feature=' Timestamp', time_scale='min', return_next_time=False):
    """
    Returns the time window starting with start_datetime index and sync_window_size min long interval
    Also returns the next datetime in the df if df's date_col column is datetime and sorted ascending
    """
    if time_scale == 'min':
        min_delta = datetime.timedelta(minutes=time_window)
    elif time_scale == 'sec':
        min_delta = datetime.timedelta(seconds=time_window)
    else:
        raise SyntaxError('time_scale should be {min} or {sec}')
    end_datetime = start_datetime + min_delta
    window = df[(df[date_feature] < end_datetime) & (df[date_feature] >= start_datetime)]
    if return_next_time:
        # Sorted assumed
        last_ind = df[df[date_feature] < end_datetime].index[-1]
        ind = np.argwhere(df.index == last_ind)[0, 0]
        next_ind = df.index[ind] if ind + 1 >= df.shape[0] else df.index[ind + 1]
        next_datetime = df[date_feature][next_ind]
        return window, end_datetime, next_datetime
    else:
        return window, end_datetime


def get_endDatetime(start_datetime, TW=2):
    "Faster for just getting end datetime"
    min_delta = datetime.timedelta(minutes=TW)
    end_datetime = start_datetime + min_delta
    return end_datetime


def append_Gfeats(df, g_data, G):
    """
    Appends graph features to dataframe. g_data is the dictionary of graph
    features returned by get_data funtion and g is the graph
    """
    node_feats = list(g_data.keys())

    for feat in node_feats:
        temp_s_sr = df.iloc[:, 1].apply(lambda node: G.nodes[node][feat])
        temp_s_df = pd.DataFrame(temp_s_sr.values, index=temp_s_sr.index, columns=['s_' + feat])

        temp_d_sr = df.iloc[:, 3].apply(lambda node: G.nodes[node][feat])
        temp_d_df = pd.DataFrame(temp_d_sr.values, index=temp_d_sr.index, columns=['d_' + feat])

        df.loc[:, 's_' + feat] = temp_s_df.values
        df.loc[:, 'd_' + feat] = temp_d_df.values

    return df.copy()


def getNewCols(df, start_datetime):
    # Returns the appended column names of the dataframe with graph features
    window, current_datetime = get_window(start_datetime, df, 1)
    G = get_Graph(window, df.columns)
    g_data, G = get_data(G)
    temp_df = append_Gfeats(window.copy(), g_data, G)
    return temp_df.columns


def appendTimeWinGFeats(df, date_feature, TW, time_scale, new_cols):
    # Appends the time windowed graph features to the whole dataset
    b_DF = pd.DataFrame(columns=new_cols)

    current_datetime = df.iloc[0][date_feature]
    last_datetime = df.iloc[-1][date_feature]
    while current_datetime <= last_datetime:
        window, current_datetime = get_window(current_datetime, df, TW, time_scale=time_scale)
        if window.empty:
            continue
        G = get_Graph(window, new_cols)
        g_data, G = get_data(G)

        temp_df = append_Gfeats(window.copy(), g_data, G)
        b_DF = b_DF.append(temp_df)
    return b_DF.copy()
