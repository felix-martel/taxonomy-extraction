import os
import math
import random
import numpy as np

from collections import defaultdict, deque, Counter
import heapq as hq
import warnings
from timer import Timer

DEFAULT_CLUSTER_DIR = "data/flat_clusters/"
DEFAULT_EMBED_FILE = "data/dbpedia/embeddings/TransE_50d_100e/ent_embeddings.npy"

def load_int_list(fname):
    with open(fname, "r") as f:
        l = [int(line.rstrip()) for line in f]
    return l

def load_mapping_list(fname):
    name2cls = {}
    cls2name = {}
    with open(fname, "r") as f:
        for line in f:
            name, cls = line.rstrip().split(" ")
            cls = int(cls)
            name2cls[name] = cls
            cls2name[cls] = name
    return name2cls, cls2name

def load_axioms_from_file(fname):
    axioms = []
    with open(fname, "r") as f:
        for line in f:
            child, parent = line.split()
            axioms.append((child, parent))
    return axioms

def load_instances(dirname):
    def filename(n): return os.path.join(dirname, "cluster.{}".format(n))
    indices = load_int_list(filename("indices"))
    labels = load_int_list(filename("labels"))
    return indices, labels

def load_classes(dirname):
    name2cls, cls2name = load_mapping_list(os.path.join(dirname, "name_to_index"))
    return name2cls, cls2name

def load_axioms(dirname):
    true_axioms = load_axioms_from_file(os.path.join(dirname, "axioms"))
    return true_axioms

def save_axioms(children, filename):    
    with open(filename, "w") as f:
        for child, parent in children:
            print(child, parent, file=f)
                
def save_class_names(name2lab, filename):    
    with open(filename, "w") as f:
        for name, label in name2lab.items():
            print(name, label, file=f)
    
def save_labelling(indices, labels, dirname):    
    i = os.path.join(dirname, "cluster.indices")
    l = os.path.join(dirname, "cluster.labels")
    with open(i, "w") as fi:
        for idx in indices:
            print(idx, file=fi)
    with open(l, "w") as fl:
        for lab in labels:
            print(lab, file=fl)
            
def save_dataset(data, dirname):
    if os.path.exists(dirname):
        x = input(f"{dirname} already exists. Are you sure you want to overwrite it? y/[n]")
        if x != "y": return
    os.makedirs(dirname, exist_ok=True)
    
    axiom_file = os.path.join(dirname, "axioms")
    clname_file = os.path.join(dirname, "name_to_index")

    save_axioms(data.axioms, axiom_file)
    save_class_names(data.name2cls, clname_file)
    save_labelling(data.indices, data.labels, dirname)

class Dataset:
    DBPEDIA_SMALL = "data/flat_clusters/"
    DBPEDIA_FULL = "data/full_flat_clusters/"
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
        indices, labels = load_instances(dirname)
        name2cls, cls2name = load_classes(dirname)
        axioms = load_axioms(dirname)
        return cls(indices=indices, labels=labels, name2cls=name2cls, cls2name=cls2name, axioms=axioms, dirname=dirname)
    
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
    