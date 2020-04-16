import numpy as np

def softmax(M, gamma=1, axis=0):
    """
    Compute the softmax over matrix M (row- or column-wise). Default is column-wise, i.e 
    return a matrix S s.t 
    S_i,j = exp(gamma * S_i,j) / sum_i' exp(gamma * S_i',j)
    Each column of S has sum 1
    """
    e = np.exp(gamma*M)
    return e / np.sum(e, axis=axis, keepdims=True)

def margin_prob(cluster, P):
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

def compute_axiom_probability(F, root, threshold=0.01, gamma=1, verbose=False):
    """
    F: F-score matrix to use for computing the type-cluster mapping probability, w/ dim (n_cluster, n_types)
    root: root cluster
    gamma: softmax parameter, gamma=0: all axioms have the same proba, gamma-->infinity: deterministic case
    threshold: probability threshold for considering that an axiom is valid
    
    # maybe we can deduce `root` from F (e.g F[-1] ?)
    """
    P = softmax(F.values, gamma=gamma)
    if verbose:
        print(np.max(P))
    #return
    dP, S = margin_prob(root, P)
    np.fill_diagonal(dP, 0.)
    
    axioms = []
    for t1, row in zip(F.columns, dP):
        for t2, proba in zip(F.columns, row):
            if proba >= threshold:
                axioms.append(((t2, t1), proba))
    if verbose:
        print(len(axioms))
    return sorted(axioms, key=lambda x: -x[1])

def extract_axioms(F, root, threshold=0.01, gamma=1, verbose=False):
    return [axiom for (axiom, _) in compute_axiom_probability(F, root, threshold=threshold, gamma=gamma, verbose=verbose)]    

def find_best_threshold(prob_axioms, true_axioms):
    """
    For each possible threshold t between 1 and 0, return:
    - precision, recall and F-score
    - number of axioms generated (ie # of axioms that have a probability >= t)
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
    """
    paxioms = compute_axiom_probability(F, clustering.root, gamma=gamma, threshold=0.)
    records = find_best_threshold(paxioms, true_axioms)
    
    t, p, r, f, n = max(records, key=lambda x:x[3])
    return gamma, t, p, r, f