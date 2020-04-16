from libs.utils import display_table
from itertools import chain
import .patterns as patterns

import numpy as np


def sum_idf(A, i):
    n_clusters, _ = A.shape
    return np.log(1 + n_clusters / A.sum(axis=0))

def root_sum_idf(A, i):
    n_clusters, _ = A.shape
    sizes = np.array([clu[cl].size for cl in cl2id])
    return np.log(1 + 1 / A[cl2id[clu.root_id]])

def depthmax_idf(A, i):
    n_clusters, _ = A.shape
    ref_depth = clu[id2cl[i]].depth
    mask = [clu[id2cl[j]].depth <= ref_depth and j != i for j in range(n_clusters)]
    return np.log(1 + 1 / A[mask].max(axis=0))

def max_idf(A, i):
    n_clusters, n_axioms = A.shape
    mask = np.arange(n_clusters) != i
    return 1 / A[mask].max(axis=0)

idfs = {
    "sum": sum_idf,
    "root_sum": root_sum_idf,
    "depthmax": depthmax_idf,
    "max": max_idf
}

def tf(A, i):
    return A[i,:]

def idf(A, i, which="depthmax"):
    return idfs[which](A, i)

def tfidf(A, i, **kwargs):
    return tf(A, i) * idf(A, i, **kwargs)


def custom_tfidf(A, i, id2ax, which_idf="depthmax", print_n=8, verbose=True, extra_values=[]):
    """This should probably move somewhere else"""
    # Compute TF-IDF
    tf = A[i,:]
    idf = idf(A, i, which=which_idf)
    tfidf = tf * idf
                  
    # Sort by decreasing order (higher TF-IDFs first)
    if verbose:
        args = np.argsort(-tfidf)[:print_n]
    
    
        display_table(
           "Axiom TF IDF TF-IDF".split(),
            extra_values,
            ([patterns.to_string(id2ax[axid]), tf[axid], idf[axid], tfidf[axid]] for axid in args)
        )
    return tfidf
    