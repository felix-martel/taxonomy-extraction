from scipy.optimize import linear_sum_assignment

from ..utils import Timer
from .common import mapping_to_axioms


def compute_mapping(F, root=None, allow_root=False, verbose=False, **kwargs):
    if not allow_root:
        # Prevent root cluster from being associated with a type
        assert root is not None, "Since you have `allow_root=False`, you must provide the root cluster"
        F.iloc[root.id] = 0.0
    
    # Compute F-matrix : Fij = F1(clu_i, cls_j)
    # F = get_F_matrix(clustering)
    # Compute a mapping m: cls -> clu that maximizes sum_j F1(m(cls_j), cls_j)
    with Timer():
        rows, cols = linear_sum_assignment(-F.values)
    int_to_cls = {i: cls for i, cls in enumerate(F.columns)}
    cls_to_clu = {int_to_cls[i]: clust for clust, i in zip(rows, cols)}
    return cls_to_clu


def extract_axioms(F, root=None, allow_root=False, verbose=False, **kwargs):    
    clu = kwargs.pop("clu")
    cls_to_clu = compute_mapping(F, root, allow_root, verbose, **kwargs)
    return mapping_to_axioms(cls_to_clu, clu)
