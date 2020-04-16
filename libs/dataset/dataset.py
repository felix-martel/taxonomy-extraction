import os
import math
import random
import numpy as np

from collections import defaultdict, deque, Counter
import heapq as hq
import warnings
from timer import Timer

from .io import load_dataset, save_dataset

DEFAULT_CLUSTER_DIR = "data/flat_clusters/"
DEFAULT_EMBED_FILE = "data/dbpedia/embeddings/TransE_50d_100e/ent_embeddings.npy"

def create_from_classes(graph, classes, class_size=100, force_size=True, **kwargs):
    """Create dataset from a list of classnames"""
    # TODO: add proper axiom creation
    cls2name = classes
    name2cls = {name: cls for cls, name in enumerate(cls2name)}
    
    if isinstance(class_size, int):
        class_size = [class_size] * len(cls2name)
    else:
        assert len(class_size) == len(cls2name), f"Size mismatch between class list ({len(cls2name)}) and size list ({len(class_size)})"
    
    indices = []
    labels = []
    for size, (t, label) in zip(class_size, name2cls.items()):
        ids = graph.sample_instances(size, from_=t, force_size=force_size, **kwargs)
        labs = [label] * len(ids)
        indices.extend(ids)
        labels.extend(labs)

    return Dataset(indices, labels, name2cls, cls2name, axioms=set())

def create_from_instances(graph, instances, is_valid=None, isa="rdf:type"):
    if is_valid is None:
        def is_valid(cls): return "dbo:" in cls
    elif isinstance(is_valid, (list, set, tuple)):
        _is_valid = set(is_valid)
        def is_valid(cls): return cls in _is_valid
    else:
        if not callable(is_valid):
            raise TypeError("'is_valid' must be a list of valid class names, a function str->bool or None")
    indices = []
    labels = []
    name2cls, cls2name = {}, {}
    r = graph.rel.to_id(isa)
    for i in instances:
        types = {t for _, _, t in graph.find_triples(h=i, r=r, as_string=True) if is_valid(t)}
        if not types:
            continue
        t = random.sample(types, 1)[0]
        if t not in name2cls:
            name2cls[t] = len(name2cls)
        t = name2cls[t]
        indices.append(i)
        labels.append(t)
    cls2name = {c:n for n, c in name2cls.items()}
    #print(len(indices), len(name2cls))
    return Dataset(indices, labels, name2cls, cls2name, axioms=set())
    
class Dataset:
    DBPEDIA_SMALL = "data/taxonomies/full_small/"
    DBPEDIA_FULL = "data/taxonomies/full_large/" #"data/full_flat_clusters/"
    REGISTERED_DATASETS = {
        "small": DBPEDIA_SMALL,
        "full": DBPEDIA_FULL,
        "large": DBPEDIA_FULL # alias for 'full'
    }
    
    def __init__(self, indices, labels, name2cls, cls2name, axioms, dirname=None, remove_empty_classes=False):
        self.ids = list(range(len(indices)))
        self.indices = indices
        self.labels = labels
        self.name2cls = name2cls
        self.cls2name = cls2name
        self.axioms = axioms
        self.class_count = self._get_class_count()
        self.class_instances = self._get_class_instances()
        self.dirname = dirname
        if remove_empty_classes:
            self.remove_empty_classes()
        
    def remove_empty_classes(self, verbose=False):
        empty_classes = {cls_name: cls_id for cls_name, cls_id in self.name2cls.items() if cls_name not in self.class_count or self.class_count[cls_name] == 0}
        if verbose:
            print(f"Remove the following empty classes: {', '.join(empty_classes)}")
        for cls_name, cls_id in empty_classes.items():
            del self.name2cls[cls_name]
            del self.cls2name[cls_id]
        self.axioms = {(a, b) for (a, b) in self.axioms if a not in empty_classes and b not in empty_classes}
            
    @classmethod
    def load(cls, dirname):
        """Load dataset for a directory"""
        if dirname in cls.REGISTERED_DATASETS:
            dirname = cls.REGISTERED_DATASETS[dirname]
        data = load_dataset(dirname)
        return cls(**data)
    
    def save(self, dirname):
        save_dataset(self, dirname)
        

    def __iter__(self):
        """Iterate over (entity id, entity class) pairs"""
        for nid, index, label in zip(self.ids, self.indices, self.labels):
            yield nid, index, label
            
    def __len__(self):
        return len(self.ids)
    
    def _get_class_count(self):
        """Get the number of entities per class"""
        return Counter(self.cls2name[cls] for cls in self.labels)
    
    def _get_class_instances(self):
        """Get the list of entity ids in each class"""
        instances = defaultdict(set)
        for instance, index, label in self:
            class_name = self.cls2name[label]
            instances[class_name].add(instance)
        return instances
    
    def sample_from_cls(self, cls_name, k=1):
        samples = random.sample(data.class_instances[cls_name], k=k)
        if k == 1:
            return samples[0]
        return samples
    
    def summary(self, n=5):
        maxlen = max(map(len, self.name2cls.keys())) + 2
        header = f"Dataset ({len(self.name2cls)} classes, {len(self)} instances):\n---\n"
        freqs = "\n".join(f"{cls:{maxlen}} {count}" for cls, count in self.class_count.most_common(n))
        return header + freqs + "\n..."
    
    def set_root(self, root):
        children, parents = zip(*self.axioms)
        subroots = set(parents) - set(children)
        if len(subroots) == 1:
            # There's alreay a single root
            return
        for child in subroots:
            if child != root:
                new_axiom = (child, root)
                if isinstance(self.axioms, set):
                    self.axioms.add(new_axiom)
                else:
                    self.axioms.append(new_axiom)
    