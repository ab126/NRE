import time
import datetime
import numpy as np


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


def get_window(start_datetime, df, date_col=' Timestamp', time_window=2, time_scale='min', return_next_time=False):
    """
    Returns the time window starting with start_datetime index and sync_window_size min long interval
    Also returns the next datetime in the df if df's date_col column is datetime and sorted ascending

    :param start_datetime: Starting datetime of the time windo
    :param df: Source DataFrame to be windowed
    :param date_col: Date column name of the DataFrame df
    :param time_window: Time window length in time_scale units
    :param time_scale: Either 'min' or 'sec'. Units of time_window datetime
    :param return_next_time: If True, returns the next flow's starting datetime after the time_window
    :return: window, end_datetime, next_datetime
        window: Window flow DataFrame
        end_datetime: Ending datetime of the time_window
        next_datetime: (Only if return_next_time==True) Next flow's starting datetime
    """
    if time_scale == 'min':
        min_delta = datetime.timedelta(minutes=time_window)
    elif time_scale == 'sec':
        min_delta = datetime.timedelta(seconds=time_window)
    else:
        raise SyntaxError('time_scale should be {min} or {sec}')
    end_datetime = start_datetime + min_delta
    window = df[(df[date_col] < end_datetime) & (df[date_col] >= start_datetime)]
    if return_next_time:
        # Sorted assumed
        last_ind = df[df[date_col] < end_datetime].index[-1]
        ind = np.argwhere(df.index == last_ind)[0, 0]
        next_ind = df.index[ind] if ind + 1 >= df.shape[0] else df.index[ind + 1]
        next_datetime = df[date_col][next_ind]
        return window, end_datetime, next_datetime
    else:
        return window, end_datetime
