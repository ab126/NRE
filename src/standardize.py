import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# TODO: Might be merged, organized

def inf_removal(df, feat_ind):
    # Ignores rows that have inf elements
    data = np.asarray(df.loc[:, feat_ind].values, dtype=float)
    inf_loc = np.where(data == np.inf)
    temp = {}
    for val in inf_loc[0]:
        temp[val] = []
    inf_loc = np.array(list(temp.keys()), dtype=np.int64)
    ind = np.isin(np.arange(data.shape[0]), inf_loc, invert=True)

    temp_df = pd.DataFrame(df.loc[ind, :])
    return temp_df.loc[:, :]


def standardize_df(df, feat_ind=None):
    # Makes the data in feat_ind zero mean & unit variance and returns it as df
    if feat_ind is None:
        feat_ind = s_df.columns
    scaler = StandardScaler()
    data = np.asarray(df.loc[:, feat_ind].values, dtype=float)
    scaler.fit(data)

    s_df = df.copy()
    s_df[feat_ind] = scaler.transform(data)
    return s_df.copy()


def appendPCA(df, feat_ind=None, n_comp=3):
    # Appends 3 pca components to the dataframe
    if feat_ind is None:
        feat_ind = df.columns
    t_df = df.copy()
    s_df = standardize_df(t_df, feat_ind)
    pca = PCA(n_components=n_comp)
    pca_result = pca.fit_transform(s_df[feat_ind].values)

    for k in range(n_comp):
        t_df['pca-' + str(k + 1)] = pca_result[:, k]
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    return t_df.copy()


def get_conType(df_conType):
    "Returns the distinct types of connections"
    con_type = {}
    for elem in df_conType:
        con_type[elem] = 0
    return list(con_type.keys())


def get_freq(df, conType):
    # Returns the dictionary of frequency of different labelled elements in df
    freq = {}
    for typ in conType:
        p = df[df['label'] == typ]
        freq[typ] = p.shape[0]
    return freq


def PCAplot(conType, s_df, title, l_save=False):
    "Form connection enumeration"
    conn_enum = {}
    plot_set = {}
    i = 0
    for typ in conType:
        p = s_df[s_df['label'] == typ]
        if p.shape[0] != 0:
            plot_set[typ] = p
            conn_enum[typ] = i
            i += 1

    plt.figure(figsize=(16, 10))
    scatter = sns.scatterplot(
        x="pca-1", y="pca-2",
        hue="label",
        palette=sns.color_palette("hls", len(conn_enum)),  # len(conn_enum)
        data=s_df,
        legend="full",
        alpha=0.9
    )

    ax = scatter.axes

    # plt.xlim(-2,2)
    # plt.ylim(-5,10)
    plt.xlabel("pca-one", fontsize=26)
    plt.ylabel("pca-two", fontsize=26)
    plt.legend(fontsize=15, markerscale=2.)

    plt.title(title, fontsize=26)
    if l_save == True:
        plt.savefig('pca_trafficOnly.png')


def attacksAtEnd(df, b_lbl_feat='b_label'):
    # For better plotting results places attacks after normal connections

    att_df = df.loc[df[b_lbl_feat] == 'attack']
    df = df.loc[df[b_lbl_feat] == 'normal']
    return df.append(att_df)
