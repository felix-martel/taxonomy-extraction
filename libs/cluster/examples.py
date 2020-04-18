import numpy as np

from ._cluster import clusterize
from libs.dataset.examples import toy_data

# Create random embeddings for the toy dataset
dim = 4
centroids = np.random.random((toy_data.n_classes, dim))
E = np.array([centroids[i]+0.05*np.random.random(dim) for i in toy_data.labels])

# Run clustering
clu = clusterize(toy_data, E)

if __name__ == "__main__":
    clu.plot()

    print("Root")
    print(clu.summary())

    ca, cb = clu.children
    print("Left")
    print(ca.summary())

    print("Right")
    print(cb.summary())
