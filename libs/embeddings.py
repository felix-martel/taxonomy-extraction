import os
import json

import itertools as it
import numpy as np
from tqdm import tqdm

from libs import CONST


DIR = CONST["embeddings"]["dirname"]  #  "data/dbpedia/embeddings"
ent_file = "ent_embeddings.npy"
rel_file = "rel_embeddings.npy"
file = {"ent": ent_file, "rel": rel_file}

TransE = [
    "TransE_50d_100e",
    "TransE_50d_140e",
]

ComplEx = [
    "ComplEx_50d_60e",
    "ComplEx_50d_10e",
    "ComplEx_50d_40e",
    "ComplEx_50d_60e",
]

DistMult = [
    "DistMult_50d_100e",
    "DistMult_50d_40e",
]

TransD = [
    "TransD_50d_50e",
]

TransH = [
    "TransH_50d_120e",
    "TransH_50d_150e",    
    "TransH_50d_50e",
]

RDF2Vec = [
    "RDF2Vec_200d_uniform",
]

HolE = [
    "HolE_50d_60e",
]

Toy = [
    "toy"
]

MODELS = {
    "ComplEx": ComplEx, "TransE": TransE, "TransH": TransH, "TransD": TransD, "RDF2Vec": RDF2Vec,
    "DistMult": DistMult, "HolE": HolE, "toy": Toy}

DEFAULT = TransE

def models(name=None):
    if name is None:
        return it.chain(*MODELS.values())
    return [m for m in models() if name in m]

def filename(name, which="ent"):
    return os.path.join(DIR, name, file[which])

def old_load(model=None, dim=None, epochs=None):
    if isinstance(model, np.ndarray):
        return model
    if model is None:
        model = DEFAULT
    if isinstance(model, list):
        m = model[0]
    else:
        m = MODELS.get(model, [model])[0]
    name = filename(m)
    return np.load(name)

def load_registry():
    with open("resources.json", "r") as f:
        r = json.load(f)
    return r.get("embeddings", dict())

def load(model=None):
    if isinstance(model, np.ndarray):
        return model
    r = load_registry()
    if model is None:
        if "default" in r:
            model = r["default"]
        else:
            raise ValueError("Since no default embeddings model is provided in config file 'resources.json', you must"
                             " provide a model name or path to function `load`.")
    model = r.get(model, model)
    return np.load(model)

def get_empty_ids(model, dim=None, epochs=None, verbose=True):
    E = load(model, dim, epochs)
    n, dim = E.shape
    return {i for i, e in tqdm(enumerate(E), total=n, desc="Entities", disable=not verbose) if not e.any()}
    

