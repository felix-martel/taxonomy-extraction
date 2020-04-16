from math import sqrt
from collections import defaultdict  
import numpy as np
#from libs.utils import euclidean
import libs.axiom
from scipy.spatial.distance import cosine, cityblock, euclidean
import scipy.spatial.distance as distance_library
from operator import itemgetter

DISTANCES = {
    "euclidean": euclidean,
    "cosine": cosine,
    "l1": cityblock    
}

def get_distance(dist):
    if isinstance(dist, str):
        if dist in DISTANCES: return DISTANCES[dist]
        return getattr(distance_library, dist)
        #if dist not in DISTANCES: raise ValueError(f"Unrecognized metric: `{dist}`. Try one of {', '.join(DISTANCES.keys())}")
        #return DISTANCES[dist]
    return dist


class SpheroidClass:
    def __init__(self, class_id, class_name, class_instances=[], parent=None, embedding_matrix=None, distance="euclidean"):
        self.id = class_id
        self.name = class_name
        self.instances = class_instances
        self.parent = parent
        self.distance = get_distance(distance)
        if self.instances and embedding_matrix is not None:
            self.centroid = self.compute_centroid(embedding_matrix)
            self.radius = self.compute_radius(embedding_matrix)
        else:
            self.centroid = None
            self.radius = 0.0
            
    @classmethod
    def build_from_dataset(cls, dataset, embeddings, distance="euclidean"):
        """Build and return a list of SpheroidClasses from a Dataset object"""
        classes = []
        instances_per_cls = defaultdict(list)
        for instance, label in zip(dataset.indices, dataset.labels):
            instances_per_cls[label].append(instance)

        for class_name, class_id in dataset.name2cls.items():
            instances = instances_per_cls[class_id]
            if not instances:
                continue
            sclass = cls(class_id, class_name, class_instances=instances, embedding_matrix=embeddings, distance=distance)
            classes.append(sclass)
        return classes          
        
    def add_instances(self, indices, labels, verbose=False):
        n_found = 0
        for instance, label in zip(indices, labels):
            if label == self.id:
                self.instances.append(instance)
                n_found += 1
        if verbose:
            print("{n_found} new instances found")
                
    def set_instances(self, indices, labels, verbose=False):
        self.instances = []
        self.add_instances(indices, labels, verbose=verbose)
        
    def compute_centroid(self, embedding_matrix):
        # self.instances shouldn't be empty
        _, dim = embedding_matrix.shape
        def vec(i): return embedding_matrix[i]
        centroid = np.zeros(dim)
        for i in self.instances:
            centroid += vec(i)
        return centroid / len(self.instances)
        
    def compute_radius(self, embedding_matrix):
        def vec(i): return embedding_matrix[i]
        #radius = sum(np.linalg.norm(vec(i)-self.centroid)**2 for i in self.instances)
        radius = sum(self.distance(vec(i), self.centroid)**2 for i in self.instances)
        return sqrt(radius / len(self.instances))
    
    def compute_spheroid(self, embedding_matrix):
        self.centroid = self.compute_centroid(embedding_matrix)
        self.radius = self.compute_radius(embedding_matrix)
        
def compare(c1, c2, distance=euclidean):
    """Check if c1 could be a subclass of c2"""
    d = distance(c1.centroid, c2.centroid)
    is_subclass = d < c2.radius and c1.radius < c2.radius
    return is_subclass, d


def compute_ristoski_spheroids(dataset, embeddings, distance="euclidean"):
    return SpheroidClass.build_from_dataset(dataset, embeddings, distance=distance)


def ristoski(classes, distance="euclidean"):
    distance = get_distance(distance)
    axioms = []
    for c1 in classes:
        a = []
        for c2 in classes:
            if c1.id == c2.id:
                continue
            is_subclass, dist = compare(c1, c2, distance)
            if is_subclass:
                axiom = (c1, c2)
                a.append((axiom, dist))
        candidate_axioms = sorted(a, key=itemgetter(1))
        if candidate_axioms:
            axiom, dist = candidate_axioms[0]
            axioms.append(axiom)
    return axioms

def build_taxonomy(dataset, embeddings, use_transitive_closure=True, distance="euclidean"):
    classes = compute_ristoski_spheroids(dataset, embeddings, distance=distance)
    axioms = ristoski(classes, distance=distance)
    axioms = set(libs.axiom.spheroids_to_axioms(axioms))        
    if use_transitive_closure:
        axioms = libs.axiom.transitive_closure(axioms)
    return axioms


def build_and_evaluate(dataset, embeddings, evaluate=True, use_transitive_closure=True, verbose=True):
    axioms = build_taxonomy(dataset, embeddings, use_transitive_closure)
    axioms_true = set(dataset.axioms)
    #if verbose:
        #print("{} raw axioms were found (before transitive closure)".format(len(axioms)))
        
    if use_transitive_closure:
        axioms = libs.axiom.transitive_closure(axioms)
        axioms_true = libs.axiom.transitive_closure(axioms_true)
    if evaluate:
        results = libs.axiom.evaluate(axioms_true, axioms, verbose=verbose)
        
    return axioms, results
    