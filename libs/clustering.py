import numpy as np
from collections import defaultdict, Counter, deque
from operator import attrgetter
from sklearn.cluster import AgglomerativeClustering
from libs.timer import Timer
import pandas as pd
from libs.metrics import f_score, precision, recall, precision_recall
from libs.pptree import print_tree
import io
from contextlib import redirect_stdout
import warnings
import random
from tqdm import tqdm

MININF = -float("inf")

def cprint(counter, min_weight=0.9, max_classes=5):
    """Display the overall composition of the cluster"""
    n = sum(counter.values())
    cum_weight = 0
    n_classes = 0
    sorted_classes = counter.most_common()
    main_classes = []
    for cls, weight in sorted_classes:
        cum_weight += weight/n
        main_classes.append((cls, weight/n))
        if cum_weight > min_weight or len(main_classes) >= max_classes:
            break

    if len(counter) > max_classes:
        main_classes.append(("other", 1-cum_weight))

    plural = "s" if n > 1 else ""        
    print("{} item{}".format(n, plural))
    for i, (name, weight) in enumerate(main_classes):
        print("{:02d} {:25.25} {:.1f}%".format(i+1, name, 100*weight))

class Cluster:
    """Represent one cluster in a hierarchical clustering tree"""
    def __init__(self, index, clu, parent=None, children=None):
        self.id = index
        self.is_root = parent is None
        self.is_leaf = children is None
        self.parent = parent
        self.children = [] if self.is_leaf else children
        if self.is_root:
            self._depth = 0
        else:
            self._depth = None
        
        if self.is_leaf:
            data = clu.data
            class_id = data.labels[index]
            class_name = data.cls2name[class_id]
            self.composition = Counter({class_name: 1})
            self.size = 1
        else:
            self.composition = sum(map(attrgetter("composition"), self.children), Counter())
            self.size = sum(map(attrgetter("size"), self.children))
            for child in children:
                child.parent = self
                
    @property
    def main_class(self):
        """Return the most frequent class in cluster"""
        class_name, count = self.composition.most_common(1)[0]
        return class_name
    
    @property
    def depth(self):
        if self._depth is None:
            self._depth = self.parent.depth + 1
        return self._depth
    
    @property
    def name(self):
        return "C<{}>".format(self.id)
        
    
    def precision(self, cls=None):
        if cls is None:
            cls = self.main_class
        p = self.composition[cls] / self.size
        return p
    
    def clipped(self, tax=None, level=-1):
        """
        Return the composition using taxonomic-equivalence of a sublevel
        
        eg Cluster{Athlete: 3, Artist: 1, Building: 1} will have :
        clipped(-1) = {Athlete: 3, Artist: 1, Building: 1}
        clipped(1) = {Person: 4, Place: 1}
        clipped(0) = {Thing: 5}
        """
        c = self.composition
        if tax is None:
            return c
        return Counter(tax[el][level].name for el in c.elements())
    
    def clipped_precision(self, cls=None, tax=None, level=-1):
        c = self.clipped(tax, level)
        if cls is None: cls= c.most_common(1)[0][0]
        return c[cls] / sum(c.values())
        
    
    def __hash__(self):
        return hash(self.id)
        
    
    def __iter__(self):
        """Iterate over all subclusters (ie children)"""
        for child in self.children:
            yield child
            
    def items(self):
        """Iterate over all entities in cluster"""
        if self.is_leaf:
            yield self.id
        else:
            for child in self:
                yield from child.items()
    
    def print(self, min_weight=0.9, max_classes=5):
        """Display the overall composition of the cluster"""
        n = self.size
        cum_weight = 0
        n_classes = 0
        sorted_classes = self.composition.most_common()
        main_classes = []
        for cls, weight in sorted_classes:
            cum_weight += weight/n
            main_classes.append((cls, weight/n))
            if cum_weight > min_weight or len(main_classes) >= max_classes:
                break

        if len(self.composition) > max_classes:
            main_classes.append(("other", 1-cum_weight))

        plural = "s" if n > 1 else ""        
        print("C<{}> ({} item{})".format(self.id, n, plural))
        for i, (name, weight) in enumerate(main_classes):
            print("{:02d} {:25.25} {:.1f}%".format(i+1, name, 100*weight))
            
    def __str__(self):    
        with io.StringIO() as buf, redirect_stdout(buf):
            self.print()
            output = buf.getvalue()
        return output[:-1]
    
    @property
    def left(self):
        if self.is_leaf:
            return None
        if len(self.children) != 2:
            warnings.warn("Cluster has {} children, so `cluster.left` and `cluster.right` are ill-defined")
        return self.children[0]
    
    @property
    def right(self):
        if self.is_leaf:
            return None
        if len(self.children) != 2:
            warnings.warn("Cluster has {} children, so `cluster.left` and `cluster.right` are ill-defined")
        return self.children[-1]
    
    
class Clustering:
    """Represent a hierarchical clustering tree (basically a wrapper around the root `Cluster`)"""
    def __init__(self, dataset, algo=AgglomerativeClustering, is_full_matrix=True, verbose=True, **params):
        self.data = dataset
        self.algo = algo
        self.params = params
        self.tree = None
        self.at_depth = None
        
        self.n_samples = len(self.data.indices)
        self.n_nodes = 2*self.n_samples - 1
        self.n_classes = len(self.data.name2cls)
        self.is_full_matrix = is_full_matrix
        
        self.root_id = self.n_nodes - 1
        self.verbose = verbose
        
    def fit(self, embeddings):
        X = embeddings
        if self.is_full_matrix:
            X = X[self.data.indices]
        with Timer(disable=not self.verbose):
            clu = self.algo(**self.params).fit(X)
        return self.set_clustering(clu)
    
    def _depth_count(self):
        at_depth = defaultdict(set)
        for cluster in self:
            at_depth[cluster.depth].add(cluster.id)
        return at_depth
        
    def set_clustering(self, clu):
        self.tree = self._clu_to_tree(clu)
        self.at_depth = self._depth_count()
        return self
        
    @property
    def root(self):
        if self.tree is not None:
            return self.tree[self.root_id]
        
    def __iter__(self):
        yield from self.tree
            
    
    def _clu_to_tree(self, clu):
        """
        Transform a 'sklearn.cluster.AgglomerativeClustering' object
        into a list of 'Cluster's, such that the i-th element contains
        cluster #i
        """
        # Create a mapping child_id --> parent_id
        revtree = {child:parent+self.n_samples 
                   for parent, children in enumerate(clu.children_) 
                   for child in children}
        tree = []
        # First, create leaves (ie cluster of size 1, one for each sample)
        for i in range(self.n_samples):
            new_node = Cluster(i, self, parent=revtree[i])
            tree.append(new_node)
        # Then create non-leaf clusters
        for i, children in enumerate(clu.children_):
            cluster_id = i + self.n_samples
            if cluster_id != self.root_id:
                parent = revtree[cluster_id]
            else:
                parent = None
            children = [tree[child_id] for child_id in children]
            new_node = Cluster(cluster_id, self, parent=parent, children=children)
            tree.append(new_node)
        return tree
    
    def sample(self, cls, k=1):
        """Return `k` leaf clusters from class `cls`"""
        sampled = [self[cid] for cid in random.sample(self.data.class_instances[cls], k)]
        if k == 1:
            return sampled[0]
        return sampled
    
    def iter_branch(self, start):
        """
        Iterate over a branch,
        ie over the path going from node 'start' to the root
        """
        if isinstance(start, int):
            # start should be a 'Cluster' object
            start = self.tree[start]
        root = self.tree[-1]
        curr_node = start
        yield curr_node
        while curr_node != root:
            curr_node = curr_node.parent
            yield curr_node
            
    def map_branch(self, node, func, *args, **kwargs):
        """Apply a function over a branch"""
        branch = []
        values = []
        for node in self.iter_branch(node):
            branch.append(node)
            values.append(func(node, *args, **kwargs))
        return branch, values
    
    def analyze_branch(self, start=None, cls=None, return_path=False, rescale=False, n_points=25):
        if start is None and cls is None:
            raise ArgumentError("You must provide at least one of `start` and `cls` argument")
        if start is None:
            start = self.sample(cls)
        if cls is None:
            cls = cluster.main_class

        p_list = []
        r_list = []
        branch = []
        
        for node in self.iter_branch(start):
            p, r = self.precision_recall(node, cls)

            p_list.append(p)
            r_list.append(r)
            branch.append(node)

        p_list = np.array(p_list)
        r_list = np.array(r_list)
        f_list = 2 * (p_list * r_list) / (p_list + r_list)
        x = np.linspace(0, 1, len(p_list))
        
        if rescale:
            x_ref = np.linspace(0, 1, n_points)
            p_list, r_list, f_list = [np.interp(x_ref, x, y) for y in (p_list, r_list, f_list)]
            x = x_ref
        if return_path:
            return x, p_list, r_list, f_list, branch
        return x, p_list, r_list, f_list
    
    def get_func_matrix(self, func, *args, **kwargs):
        classes = self.data.name2cls.keys()
        n_classes = len(classes)
        n_clusters = self.n_nodes
        
        progress_bar = tqdm
        if "verbose" in kwargs:
            verbose = kwargs.pop("verbose")
            if not verbose:
                def progress_bar(x, *args, **kwargs):
                    return x
        
        #print("Building matrix with dim: {}, {}".format(n_clusters, n_classes))
        F = np.zeros((n_clusters, n_classes))
        for j, cls in progress_bar(enumerate(classes), total=n_classes, desc="Building matrix"):
            for i, node in enumerate(self.tree):
                F[i, j] = func(node, cls, *args, **kwargs)
        return pd.DataFrame(F, columns=classes, index=range(n_clusters))
    
    def precision(self, cluster, cls=None):
        cluster = self[cluster]
        if cls is None:
            cls = cluster.main_class
        return precision(cluster.composition, cls)
    
    def recall(self, cluster, cls=None):
        cluster = self[cluster]
        if cls is None:
            cls = cluster.main_class
        return recall(cluster.composition, cls, self.data.class_count)
    
    def precision_recall(self, *args, **kwargs):
        return self.precision(*args, **kwargs), self.recall(*args, **kwargs)
    
    def f1(self, cluster, cls=None):
        cluster = self[cluster]
        if cls is None:
            cls = cluster.main_class
        return f_score(cluster, cls, self.data.class_count)
    
    def __getitem__(self, key):
        """Get cluster with id `key`. If `key` is already a cluster, return it unchanged"""
        if isinstance(key, Cluster):
            return key
        return self.tree[key]
    
    def dfs(self, start=None, max_depth=float("inf"), max_nodes=float("inf")):
        """Depth-first search over the clustering tree"""        
        if start is None: 
            start = self.root
        else:
            start = self[start]
        curr_depth = 0
        curr_nodes = 0
        unvisited = [start]
        while unvisited and curr_nodes < max_nodes:
            node = unvisited.pop()
            curr_nodes += 1
            yield node
            if node.is_leaf or node.depth >= max_depth:
                continue
            for children in node:
                unvisited.append(children)
                
    def bfs(self, start=None, max_depth=float("inf"), max_nodes=float("inf")):
        """Breadth-first search over the clustering tree"""
        if start is None: 
            start = self.root
        else:
            start = self[start]
        curr_depth = 0
        curr_nodes = 0
        unvisited = deque([start])
        while unvisited and curr_nodes < max_nodes:
            node = unvisited.popleft()
            curr_nodes += 1
            yield node
            if node.is_leaf or node.depth >= max_depth:
                continue
            for children in node:
                unvisited.append(children)

    def F(self, verbose=True):
        """
        Compute F-matrix, ie Fij := F1(clu_i, cls_j)
        
        Where F1(clu, cls) = 2*p*r/(p + r) and
        p(clu, cls) = % of items in clu with type cls
        r(clu, cls) = % of items from cls that are in clu
        
        $F_{i,j} = F1(c_i, t_j)$
        """
        return self.get_func_matrix(f_score, class_counts=self.data.class_count, verbose=verbose)
    
    def print(self, start=None, max_depth=5, prec_threshold=0.9):
        self.custom_print(start=start, max_depth=max_depth, prec_threshold=prec_threshold)
#        def halting_func(cluster):
#            return self.precision(cluster) > prec_threshold
#        def print_cluster(cluster):
#            if halting_func(cluster):
#                return "{} {}".format(cluster.name, cluster.main_class)
#            return cluster.name
#        if start is None:
#            start = self.root
#        else:
#            start = self[start]
#        print_tree(start, max_depth=max_depth, halting_func=halting_func, nameattr=print_cluster)
#        
    def custom_print(self, start=None, max_depth=5, prec_threshold=0.9, halting_func=None, print_cluster=None):
        if halting_func is None:
            def halting_func(cluster):
                return self.precision(cluster) > prec_threshold
        if print_cluster is None:
            def print_cluster(cluster):
                if halting_func(cluster):
                    return "{} {}".format(cluster.name, cluster.main_class)
                return cluster.name
        if start is None:
            start = self.root
        else:
            start = self[start]
        print_tree(start, max_depth=max_depth, halting_func=halting_func, nameattr=print_cluster)
        
    
        
        
        
        
        
    
    
                
            
    