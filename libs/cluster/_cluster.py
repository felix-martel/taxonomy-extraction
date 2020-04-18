from collections import Counter
from typing import Union, List, Optional, Callable
from operator import attrgetter

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering

import libs.embeddings
from libs.dataset import Dataset
from libs.tree import Node
from libs.utils.timer import Timer
from libs.utils.format import format_counter


def run_clustering(data: Dataset, embeddings: np.ndarray, use_full_matrix: bool = False, verbose: bool = False,
                   **params) -> AgglomerativeClustering:
    """
    Run an hierarchical clustering algorithm on a dataset. Hierarchical clustering is done using
    `sklearn.cluster.AgglomerativeClustering`.

    :param data: a `Dataset` of length n containing the ids of the entities to cluster
    :param embeddings: a m x d matrix containing the entities' embeddings
    :param use_full_matrix: if True, then m=d and embedding indices should match dataset indices. Else, m > n
        and entity i in the dataset is represented by line i in the embedding matrix
    :param verbose: whether to display a Timer
    :param params: a parameter dict passed to `AgglomerativeClustering`
    :return: the fitted AgglomerativeClustering object
    """
    if not use_full_matrix:
        embeddings = embeddings[data.indices]

    with Timer(f"{len(data)} entities with dimension {embeddings.shape[1]} clustered in {{}}", disable=not verbose):
        clu = AgglomerativeClustering(**params).fit(embeddings)
    return clu


def build_clustering(clu: AgglomerativeClustering, data: Dataset) -> "Cluster":
    """
    Build a clustering tree from a fitted AgglomerativeClustering object
    """
    n_samples: int = len(clu.children_) + 1
    edges = [(child, parent + n_samples)
             for parent, children in enumerate(clu.children_)
             for child in children]
    clustering_tree = Cluster.from_edges(edges, data=data)
    clustering_tree.init_composition()
    return clustering_tree


def clusterize(data: Dataset, embeddings: Optional[np.ndarray] = None, **params) -> "Cluster":
    """
    Run the clustering step on a given Dataset, using graph embeddings. If embeddings is set to None (default), the
    default embedding model is used, as specified in `libs.embeddings.DEFAULT`. Optional parameters can be passed to
    `run_clustering`.
    Return a clustering tree (`Cluster`)
    """
    embeddings = libs.embeddings.load(embeddings)
    clu = run_clustering(data, embeddings, **params)
    tree = build_clustering(clu, data)
    return tree


class Cluster(Node):
    """
    To implement:
    - get func matrix / cluster.F()
    Done
    - a building function
    - iterate over instances
    - summary
    - iterate over subclusters --> NO: use self.children
    - bfs / dfs : already implemented
    """

    def __init__(self, id_: int, parent: Union[None, "Cluster"] = None, tree=None, data: Union[None, Dataset] = None):
        name = f"C<{id_}>"
        super().__init__(id_, name, parent, tree)
        self.id = id_
        if data is None and self.parent is not None:
            data = self.parent.data
        self.children: List["Cluster"]
        self.data = data
        self.size = 1 if self.is_leaf else \
            sum(map(attrgetter("size"), self.children))
        self._composition: Optional[Counter] = None

    def get_composition(self) -> Counter:
        if self.data is None:
            raise ValueError("Can't compute composition if 'data' is not set")
        if self.is_leaf:
            class_id = self.data.labels[self.id]
            class_name: str = self.data.cls2name[class_id]
            return Counter({class_name: 1})
        return sum(map(attrgetter("composition"), self.children), Counter())

    def init_composition(self):
        self._composition = self.get_composition()

    @property
    def composition(self) -> Counter:
        if self._composition is None:
            self._composition = self.get_composition()
        return self._composition

    @property
    def main_class(self) -> str:
        """Return the most frequent class in cluster"""
        cls, count = self.composition.most_common(1)[0]
        return cls

    def summary(self, **params) -> str:
        """Return a summary of the cluster composition"""
        header = f"{self.name} (depth={self.depth}, size={self.size})\n"
        return header + format_counter(self.composition, **params)

    def items(self, return_ids=True):
        """Iterate over all entities in cluster

        return_ids: whether to return entity identifiers (True) or entity indices
        in the 'dataset' (False)
        Note that identifier = data.indices[index]
        """
        if self.is_leaf:
            if return_ids:
                yield self.data.indices[self.id]
            else:
                yield self.id
        else:
            for child in self.children:
                yield from child.items(return_ids)

    def get_func_matrix(self, func: Callable[..., float], *args, verbose=False, **kwargs) -> pd.DataFrame:
        classes = self.data.name2cls.keys()
        n_classes = len(classes)
        n_clusters = len(self.tree)

        X = np.zeros((n_clusters, n_classes))
        for j, cls in tqdm(enumerate(classes), total=n_classes, disable=not verbose, desc="Building matrix"):
            for i, node in enumerate(self):
                X[i, j] = func(node, cls, *args, **kwargs)

        return pd.DataFrame(X, columns=classes, index=range(n_clusters))
