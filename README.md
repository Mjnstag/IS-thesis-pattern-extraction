# thesis-pattern-extraction

overleaf link: https://www.overleaf.com/read/kjhnvsgsjzqr

## Multiprocessed versions

The multiprocessed versions of some of the notebooks related to creating graphs are located in the folder "multiprocessed files". Care should be taken to set a limit to the amount of resources, as otherwise all available cores will be used. 

## Project structure

```
├── clustering forecasting.ipynb    # Clustering and forecasting algorithm file.
├── result processing.ipynb         # File for processing the clustering forecast results.
├── elbow - silhouette.ipynb        # File for calculating elbow and silhouette scores for different amount of clusters.
├── cluster silhouette.ipynb        # File for creating silhouette plots of clusters.
├── stability.ipynb                 # File for calculating stability (similarity) of cluster indices.
├── options.txt                     # Set file locations to be used.
├──┬ multiprocessed files
|  ├── MP_cluster silhouette.ipynb  # Basic notebook for running multiproccesed version of "cluster silhouette.ipynb".
|  ├── MP_cluster_silhouette_ex.py  # File with functions needed for "MP_cluster silhouette.ipynb"
|  ├── MP_elbow - silhouette.ipynb  # Basic notebook for running multiproccesed version of "elbow - silhouette.ipynb".
|  └── MP_elbow_silhouette_ex.py    # File with functions needed for "MP_elbow - silhouette.ipynb"
└── README.md
```
