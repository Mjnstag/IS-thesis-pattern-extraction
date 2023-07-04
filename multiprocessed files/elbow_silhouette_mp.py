
# %% [markdown]
# # Imports

# %%
import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
import warnings
import time
from multiprocessing.pool import Pool
from IPython.display import display
import matplotlib
import io
matplotlib.use('Agg')

warnings.filterwarnings("ignore")

# %% [markdown]


def create_groups(data):
    # copy data to avoid changing the original data
    data_copy = data.copy()

    # group data by day
    groups = data_copy.groupby(pd.Grouper(freq='D'))

    # get the calender date of the groups
    days = list(groups.first().index.strftime('%Y:%m:%d'))

    # create a list of dataframes for each day
    gro = [groups.get_group(x).reset_index(drop=True) for x in groups.groups]

    # create a single dataframe with all days as columns
    temp = pd.concat(gro, axis=1, keys=days)

    # set index to hours and minutes
    temp.index = pd.date_range("00:00", "23:59", freq="1min").strftime('%H:%M')

    # drop all columns of temp dataframe which contain nan values
    temp.dropna(axis=1, how='any', inplace=True)

    # reduce data to every 10 minutes
    temp = temp[::10]
    # return transformed data
    return temp

# %%


def scale_data(data):
    # copy data to avoid changing the original data
    data_copy = data.copy()

    # create train and test set based on train_percentage
    train_percentage = 0.8
    train_size = int(len(data_copy.columns) * train_percentage)

    train = data_copy.iloc[:, :train_size]
    test = data_copy.iloc[:, train_size:]

    # create scaler object
    scaler = MinMaxScaler(feature_range=(0, 1))

    # fit and transform scaler to train data
    scaled_list_train = [train[col] for col in train]
    scaled_list_train = scaler.fit_transform(scaled_list_train)

    # transform test data
    scaled_list_test = [test[col] for col in test]
    scaled_list_test = scaler.transform(scaled_list_test)

    return scaled_list_train, scaled_list_test

# %%


def create_pca(data):
    # copy data to avoid changing the original data
    data_copy = data.copy()

    pca = PCA(n_components=0.85, svd_solver='full')

    # Fit and transform data
    pca_features = pca.fit_transform(data_copy)

    return pca_features

# %%


def kmeans_sillouette(data):
    data_copy = data.copy()
    wcss = []
    silhouette_scores = []

    # Calculate WCSS and silhouette scores for 1 to 10 clusters
    for i in range(1, 10):
        kmeans_pca = TimeSeriesKMeans(
            n_clusters=i, metric="dtw", n_jobs=-1).fit(data_copy)
        wcss.append(kmeans_pca.inertia_)
        try:
            silhouette_scores.append(silhouette_score(
                data_copy, kmeans_pca.labels_))
        except:
            silhouette_scores.append(0)
    return wcss, silhouette_scores

# %%


def plot_scores(column, wcss, silhouette_scores, n_cluster):
    # Plot WCSS and silhouette scores
    fig, ax1 = plt.subplots()
    x_scale = range(1, 10)

    # add WCSS data to the plt
    color = 'tab:red'
    ax1.set_xlabel('Number of clusters')
    ax1.set_ylabel('WCSS', color=color)
    ax1.plot(x_scale, wcss, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # add second y-axis
    ax2 = ax1.twinx()

    # add silhouette scores to the plt
    color = 'tab:blue'
    ax2.set_ylabel('Silhouette score', color=color)
    ax2.plot(x_scale, silhouette_scores, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    plt.title(column)

    # add vertical line for chosen number of clusters
    plt.axvline(x=n_cluster, color='r',
                label='axvline - full height', linestyle="dashed")

    fig.tight_layout()

    b = io.BytesIO()
    plt.savefig(b, format='png')
    plt.close()

    b.seek(0)
    return b.read()

# %%


def elbow(column, col_name, n_cluster):
    '''Driver function to call all other functions in order'''

    grouped_data = create_groups(column.copy())

    scaled_list_train, scaled_list_test = scale_data(grouped_data)

    pca_data = create_pca(scaled_list_train)

    wcss, silhouette_scores = kmeans_sillouette(pca_data)

    fig = plot_scores(col_name, wcss, silhouette_scores, n_cluster)
    return fig


# %%


# calls the driver function for each column in the dataframe
# in combination with the appropriate number of clusters

def main(data, clusters, processes=None):

    RUG = data.copy()

    start_time = time.perf_counter()
    figures = []
    with Pool(processes=processes) as pool:
        cols = [RUG[[col]] for col in RUG.columns]
        col_names = RUG.columns
        to_iter = list(zip(cols, col_names, clusters))
        for result in pool.starmap(elbow, to_iter):
            figures.append(result)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print("Elapsed time (s):", round(elapsed_time, 1))

    return figures
