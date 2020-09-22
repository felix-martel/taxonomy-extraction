import numpy as np

from .common import mapping_to_axioms


def compute_mapping(F, root=None, allow_root=False, verbose=False, **kwargs):
    if not allow_root:
        # Prevent root cluster from being associated with a type
        assert root is not None, "Since you have `allow_root=False`, you must provide the root cluster"
        F.iloc[root.id] = 0.0
    
    n_clusters, n_classes = F.shape
    int_to_cls = {i: cls for i, cls in enumerate(F.columns)}
    selected_clusters = set()
    cls_index_to_id = {clidx: clnum for clidx, clnum in enumerate(F.index)}
    cls_to_clu = {}
    n_ties = 0
    for cls in F.columns:
        scores = F[cls].values
        ranked_clusters = np.argsort(-scores)
        max_f = max(scores)
        if verbose: print("max F({}, *) = {:.3f}".format(cls, max_f))
        for i in ranked_clusters:
            if verbose: print("F({}, {}) = {}".format(cls, i, scores[i]))
            if cls_index_to_id[i] not in selected_clusters:
                break
        if ranked_clusters[0] != i:
            n_ties += 1
        mapped_clu = cls_index_to_id[i]
        selected_clusters.add(mapped_clu)
        cls_to_clu[cls] = mapped_clu
    print("Mapping computed. {} ties out of {} classes".format(n_ties, n_classes))
    return cls_to_clu


def extract_axioms(F, root=None, allow_root=False, verbose=False, **kwargs):
    clu = kwargs.pop("clu")
    cls_to_clu = compute_mapping(F, root, allow_root, verbose, **kwargs)
    return mapping_to_axioms(cls_to_clu, clu)
