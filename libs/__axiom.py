from warnings import warn


warn("Module axiom.py is deprecated. Use 'libs.taxonomy.evaluation' instead.", category=DeprecationWarning)


def pprint(axiom):
    c1, c2 = axiom
    print("{} ⊑ {}".format(c1, c2))
    
def safe_divide(p, q):
    if q == 0: return 0.0
    return p / q
    
def spheroids_to_axioms(sph_axioms):
    """Convert a list of pairs (SpheroidClass, SpheroidClass) to a list of pairs (classname, classname)"""
    return [(c1.name, c2.name) for c1, c2 in sph_axioms]
    
    
def evaluate(axioms_true, axioms_pred, verbose=True):
    n_tp = len(axioms_pred & axioms_true)

    prec = safe_divide(len(axioms_pred & axioms_true), len(axioms_pred))
    rec = safe_divide(len(axioms_pred & axioms_true), len(axioms_true))
    f1 = safe_divide(2 * (prec * rec), (prec + rec))

    if verbose:
        print("precision={:.4f}%".format(100*prec))
        print("recall={:.4f}%".format(100*rec))
        print("F-score={:.4f}%\n".format(100*f1))
    return prec, rec, f1

def evaluate_full(axioms_true, axioms_pred, verbose=True):
    p1, r1, c1 = evaluate(axioms_true, axioms_pred, False)
    axioms_true = transitive_closure(axioms_true)
    axioms_pred =transitive_closure(axioms_pred)
    p2, r2, c2 = evaluate(axioms_true, axioms_pred, False)
    
    if verbose:
        print("""
closure  \tno    \t\tyes
---------------------------------------
precision\t{:.2f}%\t\t{:.2f}%
recall   \t{:.2f}%\t\t{:.2f}%
f1       \t{:.2f}%\t\t{:.2f}%
""".format(*[100*x for x in (p1, p2, r1, r2, c1, c2)]))
    return (p1, r1, c1), (p2, r2, c2)

def transitive_closure(axioms):
    _c = set(x[0] for x in axioms)
    _p = set(x[1] for x in axioms)
    roots = _p -_c
    if not roots:
        raise ValueError("Taxonomy contains a cycle")

    c2p = {c:p for c,p in axioms}

    new_axioms = []
    for c in c2p:
        p = c
        while p in _c:
            p = c2p[p]
            new_axioms.append((c,p))

    
    return set(new_axioms) | set(axioms)