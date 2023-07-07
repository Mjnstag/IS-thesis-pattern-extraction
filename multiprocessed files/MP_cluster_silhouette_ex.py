# %% [markdown]
# # Imports

# %%
import warnings
from tslearn.clustering import TimeSeriesKMeans
from sklearn.preprocessing import MinMaxScaler
import time
from multiprocessing.pool import Pool
from IPython.display import display
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import io
matplotlib.use('Agg')


# Preprocessing
# Algorithms


warnings.filterwarnings("ignore")


# %%

def create_groups(data):
    data_copy = data.copy()
    groups = data_copy.groupby(pd.Grouper(freq='D'))

    # get the calender date of the groups
    days = list(groups.first().index.strftime('%Y:%m:%d'))

    gro = [groups.get_group(x).reset_index(drop=True) for x in groups.groups]

    temp = pd.concat(gro, axis=1, keys=days)

    temp.index = pd.date_range("00:00", "23:59", freq="1min").strftime('%H:%M')

    # drop all columns of temp dataframe which contain nan values
    temp.dropna(axis=1, how='any', inplace=True)

    return temp[::10]

# %%


def scale_data(data):
    data_copy = data.copy()
    train_percentage = 0.8
    train_size = int(len(data_copy.columns) * train_percentage)

    train = data_copy.iloc[:, :train_size]
    test = data_copy.iloc[:, train_size:]

    scaler = MinMaxScaler(feature_range=(0, 1))

    scaled_list_train = [train[col] for col in train]
    scaled_list_train = scaler.fit_transform(scaled_list_train)

    scaled_list_test = [test[col] for col in test]
    scaled_list_test = scaler.transform(scaled_list_test)

    return scaled_list_train, scaled_list_test

# %%


def create_pca(data):
    data_copy = data.copy()

    pca = PCA(n_components=0.85, svd_solver='full')

    # Fit and transform data
    pca_features = pca.fit_transform(data_copy)

    return pca_features

# %%


def create_kmeans(pca_data, scaled_train, scaled_test, clusters=4):
    temp_pca_data = pca_data.copy()
    temp_scaled_train = scaled_train.copy()
    temp_scaled_test = scaled_test.copy()

    kmeans_pca = TimeSeriesKMeans(
        n_clusters=clusters, metric="dtw").fit(temp_pca_data)
    train_pca_features = kmeans_pca.labels_
    test_pca_features = kmeans_pca.predict(temp_scaled_test)

    return train_pca_features, test_pca_features

# %%


def plot_scores(scaled_list_train, train_lab, column):
    fig, ax = plt.subplots((len(set(train_lab))))
    fig.suptitle(column)
    for pos, label in enumerate(set(train_lab)):
        values = scaled_list_train[(train_lab == label).nonzero()[0]]
        for value in values:
            ax[pos].plot(value, c="gray", alpha=0.4)
        ax[pos].plot(np.average(values, axis=0), c="red")

    for i, ax in enumerate(ax.ravel()):  # 2
        ax.set_title("Cluster {}".format(i))  # 3

    fig.tight_layout()
    # plt.show()
    b = io.BytesIO()
    plt.savefig(b, format='png')
    plt.close()

    b.seek(0)
    return b.read()

# %%


def average_cluster(column, col_name, n_cluster):
    grouped_data = create_groups(column.copy())

    scaled_list_train, scaled_list_test = scale_data(grouped_data)

    pca_features = create_pca(scaled_list_train)

    train_lab, test_lab = create_kmeans(
        pca_features, scaled_list_train, scaled_list_test, n_cluster)

    fig = plot_scores(scaled_list_train, train_lab, col_name)
    return fig
# %%


def main(data, clusters, processes=None):

    RUG = data.copy()

    start_time = time.perf_counter()
    figures = []
    with Pool(processes=processes) as pool:
        cols = [RUG[[col]] for col in RUG.columns]
        col_names = RUG.columns
        to_iter = list(zip(cols, col_names, clusters))
        for result in pool.starmap(average_cluster, to_iter):
            figures.append(result)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print("Elapsed time (s):", round(elapsed_time, 1))

    return figures


# %%
