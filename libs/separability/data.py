import numpy as np
import os
from sklearn.model_selection import train_test_split
from libs.dataset import DEFAULT_EMBED_FILE
from collections import defaultdict, Counter
from tqdm import tqdm


DEFAULT = "_default_"

def load_embeddings(model=DEFAULT):
    if isinstance(model, np.ndarray): 
        return model
    if model == DEFAULT:
        filename = DEFAULT_EMBED_FILE
    else:
        model_name = "data/dbpedia/embeddings/{}/ent_embeddings.npy"
        filename = model_name.format(model)
    assert os.path.exists(filename)
    return np.load(filename)

def build_training_set(instances_a, instances_b, model=DEFAULT, **kwargs):
    na, nb = len(instances_a), len(instances_b)
    indices = [*instances_a, *instances_b]    

    y = np.concatenate([np.zeros(na), np.ones(nb)])
    X = load_embeddings(model)[indices]
    X_train, X_test, y_train, y_test = train_test_split(X, y, **kwargs)
    return X_train, X_test, y_train, y_test

def get_centroids(graph, valid_types=None, embeddings=DEFAULT, type_rel="rdf:type", verbose=True):
    if callable(valid_types):
        is_valid = valid_types
    elif valid_types is None:
        is_valid = lambda x:True
    else:
        is_valid = lambda x: x in valid_types
    
    if isinstance(embeddings, str):
        embeddings = load_embeddings(embeddings)
    _, dim = embeddings.shape
    
    isa = graph.rel.to_id(type_rel)
    types = graph._r[isa]
    centroids = defaultdict(lambda:np.zeros(dim))
    counts = Counter()
    for h, ts in tqdm(types.items(), total=len(types), disable=not verbose, desc="Computing centroids"):
        for t in ts:
            t = graph.ent.to_name(t)
            if is_valid(t):
                centroids[t] += embeddings[h]
                counts[t] += 1

    for t, c in counts.items():
        centroids[t] /= c

    return centroids, counts

