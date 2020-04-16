import abc
from enum import Enum
from .patterns import extract_types_from_cluster
import numpy as np
### DISPLAY UTILITIES ###
import libs.table as libtab

from itertools import zip_longest
def tag_line(line, tag): return [f"<{tag}>{x}</{tag}>" for x in line]
def bold_line(line): return tag_line(line, "b")
def italic_line(line): return tag_line(line, "i")

def join(left, right, columns=None):
    left = list(left)
    right = list(right)
    n = len(left[0] if left else right[0])
    
    header = [
        bold_line(["Left"] + (n-1)*[""] + ["Right"])
    ]
    if columns is not None: 
        columns = columns + [""] * (n - len(columns))
        header.append(italic_line(2 * columns))
        
    joined_table = header + [[*l, *r] for l, r in zip_longest(left, right, fillvalue=("",)*n)]
    return joined_table
def create_section(a, tab):
    libtab.print_html(a, tag="b")
    libtab.display_table(tab)
### END OF DISP. UTILS ###

class ClusterModule(abc.ABC):
    def __init__(self, clustering, graph=None, data=None, **kwargs):
        self.clu = clustering
        self.graph = graph
        self.data = data
        self.kwargs= kwargs
        
    @abc.abstractmethod
    def to_table(self, cluid, **kwargs):
        pass
    

class KnownTypeModule(ClusterModule):
    def to_table(self, cluid, k=5):        
        cluster = self.clu[cluid]
        n = cluster.size
        return [(cls.replace("dbo:", ""), 100*count/n) for cls, count in cluster.composition.most_common(k)]
    

class TrueTypeModule(ClusterModule):
    """
    Returns the most frequent types in a cluster (with access to the full KG, not only to the training set)
    """
    def to_table(self, cluid, k=None, **kwargs):
        params = {**self.kwargs, **kwargs}
        type_count = extract_types_from_cluster(self.clu[cluid], self.graph, self.data, **params)
        return [
            (cls.replace("dbo:", ""), 100*freq) for cls, freq in type_count.most_common(k)
        ]
    

class RankingModule(ClusterModule):
    """
    Return the types $t_1, \ldots, t_n$ for which the input cluster $C$ is a good cluster, i.e types such that $F(C, t)$ is high

    If $F(c, t) = \max_c F(c, t)$, then $\texttt{get_ranks}(c)$ will contain $t$
    """
    def __init__(self, clustering, graph=None, data=None, F=None, **kwargs):
        super().__init__(clustering, graph, data, **kwargs)
        if F is None:
            F = self.clu.F()
        self.F = F
        self.F_ranked = np.argsort(-F.values, axis=0).argsort(axis=0)
        self.col_to_ind = {cls: i for i, cls in enumerate(self.clu.data.name2cls.keys())}
        self.ind_to_col = {i: cls for cls, i in self.col_to_ind.items()} 
        
    def to_table(self, cluid, n_best=5, max_rank=20):
        """
        Each line contains a type, with the rank and the F-score of cluster cluid. 
        """
        results = []
        for k in np.argsort(self.F_ranked[cluid])[:n_best]:
            cls = self.ind_to_col[k]
            rank = self.F_ranked[cluid, k]
            f_score = self.F.values[cluid, k]
            if rank > max_rank:
                return results
            results.append((cls, rank, f_score))
        return results

        