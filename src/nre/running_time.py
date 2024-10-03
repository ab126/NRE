import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression


# TODO: Finish this module

def slice_runtime_df(df_run_time, col_name, col_val, col_sweep):
    """Given the run time dataframe of runtime experiments, gets the values of the desired column
    given by col_name and col_val and returns results for the parameter col_sweep"""

    t_df = df_run_time[df_run_time[col_name] == col_val].copy()
    tt_df = pd.DataFrame()
    for _, row in t_df.iterrows():
        temp_t_sim = row['t_sim']
        dt_sim = [temp_t_sim[i + 1] - temp_t_sim[i] for i in range(len(temp_t_sim) - 1)]
        temp_df = pd.DataFrame({col_sweep: row[col_sweep], 'dt_sim': dt_sim, 'Method': row['Method']})
        tt_df = pd.concat((tt_df, temp_df), ignore_index=True)
    tt_df[col_sweep] = tt_df[col_sweep].astype(str)
    return tt_df


def fit_lines_runtime_df(tt_df, col_sweep):
    """Fits a line to running times wrt the col_sweep"""
    y_lines = []
    x_lines = []
    for mtd in np.unique(tt_df['Method']):
        x = tt_df[tt_df['Method'] == mtd][col_sweep].values.astype(float).reshape((-1, 1))
        y = tt_df[tt_df['Method'] == mtd]['dt_sim'].values

        model = LinearRegression().fit(x, y)
        x_line = np.unique(tt_df[col_sweep]).astype(float)
        y_line = model.predict(x_line.reshape((-1, 1)))
        y_lines.append(y_line)
        #x_lines.append(map(str, x_line))
        x_lines.append(x_line.astype(str))  # Boxplot only accepts categorical variables
    return x_lines, y_lines
