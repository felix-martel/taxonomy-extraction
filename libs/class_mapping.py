import os

from scipy.optimize import linear_sum_assignment
from libs.timer import Timer
from libs.metrics import f_score
import numpy as np


def get_F_matrix(clustering):
    return clustering.get_func_matrix(f_score, class_counts=clustering.data.class_count)

def softmax(M, gamma=1, axis=0):
    e = np.exp(gamma*M)
    return e / np.sum(e, axis=axis, keepdims=True)

class ClassClusterMapping:
    def __init__(self, clustering, verbose=False, **kwargs):
        self.clu = clustering
        self.verbose = verbose
        self.cls_to_clu = self._compute_mapping(clustering, **kwargs)
        
    def _compute_mapping(self, clustering, **kwargs):
        pass
    
    def print(self, *args, **kwargs):
        if self.verbose: print(*args, **kwargs)
    
    def save(self, filename):
        dirname = os.path.split(filename)[0]
        os.makedirs(dirname, exist_ok=True)

        with open(filename, "w") as f:
            for classname, clust in self.cls_to_clu.items():
                print(classname, clust, file=f)
        self.print("class-->cluster mapping saved in <{}>".format(filename))
        
    def old_to_axioms(self, override_root=True, root_class="owl:Thing"):
        if override_root:
            assert self.clu.root_id not in self.cls_to_clu.values()
            self.cls_to_clu[root_class] = self.clu.root_id

        selected_clusters = {clu:cls for cls, clu in self.cls_to_clu.items()}   
        axioms = []
        for child_name, cluster in self.cls_to_clu.items():
            node = self.clu.tree[cluster]
            if node.is_root:
                continue
            parent = node.parent
            while parent.id not in selected_clusters:
                parent = parent.parent
                if parent.is_root:
                    break
            parent_name = selected_clusters[parent.id]
            axioms.append((child_name, parent_name))
        return set(axioms)
        
    def to_axioms(self, override_root=True, root_class="owl:Thing"):
        if override_root:
            assert self.clu.root_id not in self.cls_to_clu.values()
            self.cls_to_clu[root_class] = self.clu.root_id
        
        selected_clusters = {clu:cls for cls, clu in self.cls_to_clu.items()}   
        axioms = []
        for child_name, cluster in self.cls_to_clu.items():
            node = self.clu.tree[cluster]
            if node.is_root:
                continue
            parent = node.parent
            while not parent.is_root:
                parent = parent.parent
                if parent.id in selected_clusters:
                    parent_name = selected_clusters[parent.id]
                    axioms.append((child_name, parent_name))
                    break
        return set(axioms)
    
class LocalMaxMapping(ClassClusterMapping):
    """Find a local optimum for each class"""
    def _compute_mapping(self, clustering, allow_root=False, **kwargs):
        F = get_F_matrix(clustering) if "F" not in kwargs else kwargs["F"]
        if not allow_root:
            # Prevent root cluster from being associated with a type
            F.iloc[clustering.root_id] = 0.0
        n_clusters, n_classes = F.shape
        int_to_cls = {i: cls for i, cls in enumerate(clustering.data.name2cls.keys())}
        selected_clusters = set()
        cls_index_to_id = {clidx: clnum for clidx, clnum in enumerate(F.index)}
        cls_to_clu = {}
        n_ties = 0
        for cls in F.columns:
            scores = F[cls].values
            ranked_clusters = np.argsort(-scores)
            max_f = max(scores)
            self.print("max F({}, *) = {:.3f}".format(cls, max_f))
            for i in ranked_clusters:
                self.print("F({}, {}) = {}".format(cls, i, scores[i]))
                if cls_index_to_id[i] not in selected_clusters:
                    break
            if ranked_clusters[0] != i:
                n_ties += 1
            mapped_clu = cls_index_to_id[i]
            selected_clusters.add(mapped_clu)
            cls_to_clu[cls] = mapped_clu
        print("Mapping computed. {} ties out of {} classes".format(n_ties, n_classes))
        return cls_to_clu
    
class CustomLocalMaxMapping(ClassClusterMapping):
    """Find a local optimum for each class"""
    def _compute_mapping(self, clustering, allow_root=False, **kwargs):
        F = kwargs["F"]
        del kwargs["F"]
        
        if not allow_root:
            # Prevent root cluster from being associated with a type
            F.iloc[clustering.root_id] = 0.0
           
        n_clusters, n_classes = F.shape
        #int_to_cls = {i: cls for i, cls in enumerate(clustering.data.name2cls.keys())}
        selected_clusters = set()
        cls_index_to_id = {clidx: clnum for clidx, clnum in enumerate(F.index)}
        cls_to_clu = {}
        n_ties = 0
        for cls in F.columns:
            scores = F[cls].values
            ranked_clusters = np.argsort(-scores)
            max_f = max(scores)
            self.print("max F({}, *) = {:.3f}".format(cls, max_f))
            for i in ranked_clusters:
                self.print("F({}, {}) = {}".format(cls, i, scores[i]))
                if cls_index_to_id[i] not in selected_clusters:
                    break
            if ranked_clusters[0] != i:
                n_ties += 1
            mapped_clu = cls_index_to_id[i]
            selected_clusters.add(mapped_clu)
            cls_to_clu[cls] = mapped_clu
        print("Mapping computed. {} ties out of {} classes".format(n_ties, n_classes))
        return cls_to_clu
        
    
class GlobalMaxMapping(ClassClusterMapping): 
    """Find a global maximum"""
    def _compute_mapping(self, clustering, allow_root=False, **kwargs):
        F = get_F_matrix(clustering) if "F" not in kwargs else kwargs["F"]        
        if not allow_root:
            # Prevent root cluster from being associated with a type
            F.iloc[clustering.root_id] = 0.0
        # Compute F-matrix : Fij = F1(clu_i, cls_j)
        #F = get_F_matrix(clustering)
        # Compute a mapping m: cls -> clu that maximizes sum_j F1(m(cls_j), cls_j)
        with Timer():
            rows, cols = linear_sum_assignment(-F.values)
        int_to_cls = {i: cls for i, cls in enumerate(clustering.data.name2cls.keys())}
        cls_to_clu = {int_to_cls[i]: clust for clust, i in zip(rows, cols)}
        return cls_to_clu
    
class CustomMapping(ClassClusterMapping):
    """Pass a custom function to compute cls_to_clu mapping"""
    def _compute_mapping(self, clustering, **kwargs):
        func = kwargs["func"]
        del kwargs["func"]
        return func(clustering, **kwargs)
    
class ProbabilisticMapping(ClassClusterMapping):
    """Uses a probabilistic version of the algorithm. See `00_Pipeline_Experiment.ipynb` for details"""
    pass
    
class RankedMaxMapping(ClassClusterMapping):
    pass
