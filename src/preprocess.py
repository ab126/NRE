import pandas as pd
import numpy as np


def convert_df(df, dtype_dict):
    """
    Adjusts the column names to feature names instead of integers and map the
    dtypes if provided
    ---------------
    :param df: Source DataFrame.
    :param dtype_dict: Dictionary of column_name : dtype that specifies the formatting dtype of each column.
    :return:
        df: Formatted DataFrame.
    """
    for col, typ in dtype_dict.items():
        df[col] = df[col].astype(typ)
    return df.copy()


def get_act_cols(df):
    """
    Returns the DataFrame with only non-repeated columns of df
    ----------------
    :param df: Source DataFrame.
    :return: Formatted DataFrame with non-repeating columns.
    """
    df_cols = df.columns
    val_seen = {}
    for val in df_cols:
        val_seen[val] = False
    actual_cols_pos = {}
    for val, i in zip(df_cols, np.arange(len(df_cols))):
        if not val_seen[val]:
            actual_cols_pos[val] = i
            val_seen[val] = True
    new_cols = np.sort(list(actual_cols_pos.values()))
    temp_df = pd.DataFrame(df.iloc[:, new_cols].values, columns=df_cols[new_cols])
    return temp_df


def preprocess_df(df_input, date_col, dtype_dict=None):
    """
    Preprocess the dataframe into a canonical form. Assigns dtypes to columns, checks for repeated columns, discards
    rows with NaN or Inf values and sorts the entries/rows by date_col.
    --------------------
    :param df_input: Raw DataFrame to be formatted.
    :param date_col: Column of df_input that will be mapped to datetime object and will be used for sorting rows.
    :param dtype_dict: (optional) Dictionary of column_name : dtype that specifies the formatting dtype of each column.
    :return:
        df: Formatted df that can be used by ConnectivityUnit object.
    """
    if dtype_dict is not None:
        df = convert_df(df_input, dtype_dict)
    else:
        df = df_input.copy()
    df = get_act_cols(df)
    df = df[~df.isnull().any(axis=1)].copy()  # Discard NaN rows
    df = df[~df.isin([np.inf, -np.inf]).any(axis=1)]  # Discard inf rows

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=[date_col])
    return df


