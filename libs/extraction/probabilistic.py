import numpy as np
import functools
from collections import defaultdict


def softmax(M, beta=1, axis=0):
    """
    Compute the softmax over matrix M (row- or column-wise). Default is column-wise, i.e 
    return a matrix S s.t 
    S_i,j = exp(beta * S_i,j) / sum_i' exp(beta * S_i',j)
    Each column of S has sum 1
    """
    e = np.exp(beta * M)
    return e / np.sum(e, axis=axis, keepdims=True)

def oldest_margin_prob(cluster, P):
    """
    Recursively compute dP and S for a given cluster.
    `cluster` is a `Cluster` c (from `libs.clustering`), and `P` is the probability vector for cluster c, i.e
    P_i is the probability that c is the cluster representing type t_i
    """
    _, n_types = P.shape
    if cluster.is_leaf:
        dP = np.zeros((n_types, n_types))
        S = P[cluster.id]
        return dP, S
    dPs, Ss = zip(*(margin_prob(child, P) for child in cluster.children))
    
    dP, S = sum(dPs), sum(Ss)
    dP += np.outer(P[cluster.id], S)
    S += P[cluster.id]
    return dP, S

def margin_prob(cluster, P):
    """
    Recursively compute dP and S for a given cluster.
    `cluster` is a `Cluster` c (from `libs.clustering`), and `P` is the probability vector for cluster c, i.e
    P_i is the probability that c is the cluster representing type t_i
    """
    _, n_types = P.shape
    def compressed_prob(cluster, P):
        if cluster.is_leaf: return np.zeros((n_types, n_types)), P[cluster.id]
        return margin_prob(cluster, P)
    if cluster.is_leaf:
        dP = np.zeros((n_types, n_types))
        S = P[cluster.id]
        return dP, S
    dPs, Ss = zip(*(compressed_prob(child, P) for child in cluster.children))
    
    dP, S = sum(dPs), sum(Ss)
    dP += np.outer(P[cluster.id], S)
    S += P[cluster.id]
    return dP, S

def optim_margin_prob(clu, P):
    """Test for computing dP with O(log(N)) complexity. Same results as `margin_prob` but less memory-intensive"""
    p_cache = dict()
    s_cache = dict()
    _, n_types = P.shape
    for cluster in reversed(list(clu.dfs())):
        if cluster.is_leaf: continue
        dP = sum(p_cache.pop(child) for child in cluster.children if not child.is_leaf)
        # Handle empty sequences (for cluster with no non-leaf child)
        if isinstance(dP, int): dP = np.zeros((n_types, n_types))
        S = sum(P[child.id] if child.is_leaf else s_cache.pop(child) for child in cluster.children)
        dP += np.outer(P[cluster.id], S)
        S += P[cluster.id]
        p_cache[cluster] = dP
        s_cache[cluster] = S
    return p_cache[clu.root]   
                
                
def compute_axiom_probability(F, root, beta=1, verbose=False):
    """
    F: F-score matrix to use for computing the type-cluster mapping probability, w/ dim (n_cluster, n_types)
    root: root cluster
    beta: softmax parameter, beta=0: all axioms have the same proba, beta-->infinity: deterministic case
    threshold: probability threshold for considering that an axiom is valid
    
    # maybe we can deduce `root` from F (e.g F[-1] ?)
    """
    P = softmax(F.values, beta=beta)
    if verbose:
        print(np.max(P))
    dP = optim_margin_prob(root, P)
    # np.fill_diagonal(dP, 0.) # Remove a⊑a axioms (technically true, but of little interest)
    return dP

def get_branches(a, revtree):
    """
    For a node `a` and a directed acyclic graph `revtree`, find all paths from `a` to a root node.
    `revtree` format: dict node --> parents(node) (parents(v) are all vertices v' such that arc v'--> v exists in graph
    """
    if not revtree[a]: return [[a]]
    all_branches = []
    for p in revtree[a]:
        branches = get_branches(p, revtree)
        for b in branches:
            all_branches.append([a, *b])
    return all_branches
    #return [[a] + get_branches(p, revtree) for p in revtree[a]]

def weight_branch(branch, weights=None):
    if weights is None or not branch:
        return len(branch)
    weight = sum(weights[(a, b)] for a,b in zip(branch[:-1], branch[1:]))
    return weight

def compress_axioms(axioms, weights=None):
    """
    Opposite of transitive closure. 
    Transform a directed acyclic graph into a tree by removing redundant edges
    `axioms` : list of subsumption axioms, ie pairs (a, b) s.t a is subclass of b, or alternatively that
    arc b --> a exists in graph
    """
    if not axioms: return set()
    children, parents = zip(*axioms)
    classes = {*children, *parents}
    revtree = defaultdict(set)
    for a, b in axioms:
        revtree[a].add(b)
    branches = dict()
    new_axioms = set()
    for a in classes:
        # Retrieve all branches from a to root
        branches = get_branches(a, revtree)
        # Keep only longest branch
        branch = sorted(branches, key=functools.partial(weight_branch, weights=weights))[-1]
        if len(branch) <= 1:
            continue
        parent = branch[1]
        new_axioms.add((a, parent))
    return new_axioms

def is_connected(u, v, tree):
    """Check wether there is a directed path from u to v"""
    if u not in tree:
        return False
    if u == v:
        return True
    unprocessed = {u}
    while unprocessed:
        u = unprocessed.pop()
        if u not in tree:
            continue
        if v in tree[u]:
            return True
        unprocessed.update(tree[u])
    return False

def build_taxonomy(F, dP, threshold=0.5, verbose=False, compress=True):
    axioms = sorted((((t2, t1), proba) for t1, row in zip(F.columns, dP)
                              for t2, proba in zip(F.columns, row)
                              if proba >= threshold and t2 != t1), key=lambda x:-x[1])
    children = defaultdict(set)
    parents = defaultdict(set)
    tree = set()
    weights = dict()
    for (a, b), p in axioms:
        if is_connected(a, b, children) or p < threshold:
            continue
        tree.add((a, b))
        weights[(a, b)] = p
        children[b].add(a)
    if compress:
        return compress_axioms(tree, weights)
    return tree

def old_build_taxonomy(F, dP, threshold=0.5, verbose=False, compress=True):
    axioms = sorted((((t2, t1), proba) for t1, row in zip(F.columns, dP)
                              for t2, proba in zip(F.columns, row)
                              if proba >= threshold and t2 != t1), key=lambda x:-x[1])
    children = defaultdict(set)
    parents = defaultdict(set)
    tree = set()
    weights = dict()
    for (a, b), p in axioms:
        if (parents[b] | {b}) & (children[a] | {a}) or p < threshold:
            continue
        tree.add((a, b))
        weights[(a, b)] = p
        new_parents = {b} | parents[b]
        new_children = {a} | children[a]
        for c in new_parents:
            children[c] |= new_children
        for c in new_children:
            parents[c] |= new_parents
        #children[b] = children[b] | {a} | children[a]
    if compress:
        return compress_axioms(tree, weights)
    return tree

def older_build_taxonomy(F, dP, threshold=0.01, verbose=False):
    axioms = sorted((((t2, t1), proba) for t1, row in zip(F.columns, dP)
                              for t2, proba in zip(F.columns, row)
                              if proba >= threshold and t1 != t2), key=lambda x:-x[1])
    tree = dict()
    def root(x):
        if x not in tree: return x
        #if x == tree[x]: raise ValueError(f"{x} is its own parent")
        return root(tree[x])
    pruned_axioms = set()
    for (a, b), p in axioms:
        r = root(b)
        if a in tree or r == a:
            continue
        tree[a] = b
        pruned_axioms.add((a, b))
        
    return pruned_axioms
        
def extract_axioms(F, root=None, threshold=0.1, beta=100, verbose=False, compress=True):
    if root is None:
        root = len(F) - 1
    dP = compute_axiom_probability(F, root, beta=beta, verbose=verbose)
    return build_taxonomy(F, dP, threshold=threshold, verbose=verbose, compress=compress)
    

def find_best_threshold(prob_axioms, true_axioms):
    """
    For each possible threshold t between 1 and 0, return:
    - precision, recall and F-score
    - number of axioms generated (ie # of axioms that have a probability >= t)
    DEPRECATED
    """
    def safe_divide(p, q):
        if q == 0: return 0.
        return p / q
    n_tp = 0
    n_p = 0
    n_t = len(true_axioms)
    curr_threshold = 1.0
    curr_prec = safe_divide(n_tp, n_p)
    curr_rec = safe_divide(n_tp, n_t)
    curr_f = curr_prec * curr_rec
    
    records = [(curr_threshold, curr_prec, curr_rec, curr_f, n_p)]
    for i, (ax, prob) in enumerate(prob_axioms):
        n_p += 1
        if ax in true_axioms: n_tp += 1
        curr_threshold = (curr_threshold + prob) / 2
        curr_prec = safe_divide(n_tp, n_p)
        curr_rec = safe_divide(n_tp, n_t)
        curr_f = safe_divide(2 * curr_prec * curr_rec, curr_prec + curr_rec)
        records.append((curr_threshold, curr_prec, curr_rec, curr_f, n_p))
    return records

def evaluate_gamma(gamma, F, clustering):
    """
    Return the max possible F-score one can get with a given gamma value
    DEPRECATED
    """
    paxioms = compute_axiom_probability(F, clustering.root, beta=gamma, threshold=0.)
    records = find_best_threshold(paxioms, true_axioms)
    
    t, p, r, f, n = max(records, key=lambda x:x[3])
    return gamma, t, p, r, f