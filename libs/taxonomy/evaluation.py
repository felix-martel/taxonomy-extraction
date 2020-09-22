from libs.utils.misc import safe_divide


def evaluate(axioms_true, axioms_pred, verbose=True):
    prec = safe_divide(len(axioms_pred & axioms_true), len(axioms_pred))
    rec = safe_divide(len(axioms_pred & axioms_true), len(axioms_true))
    f1 = safe_divide(2 * (prec * rec), (prec + rec))

    if verbose:
        print("precision={:.4f}%".format(100 * prec))
        print("recall={:.4f}%".format(100 * rec))
        print("F-score={:.4f}%\n".format(100 * f1))
    return prec, rec, f1


def evaluate_full(axioms_true, axioms_pred, verbose=True, sep="."):
    axioms_pred = set(axioms_pred)
    axioms_true = set(axioms_true)
    p1, r1, c1 = evaluate(axioms_true, axioms_pred, False)
    axioms_true = transitive_closure(axioms_true)
    axioms_pred = transitive_closure(axioms_pred)
    p2, r2, c2 = evaluate(axioms_true, axioms_pred, False)

    if verbose:
        print("""
closure  \tno    \t\tyes
---------------------------------------
precision\t{:.2f}%\t\t{:.2f}%
recall   \t{:.2f}%\t\t{:.2f}%
f1       \t{:.2f}%\t\t{:.2f}%
""".format(*[100 * x for x in (p1, p2, r1, r2, c1, c2)]).replace(".", sep))
    return (p1, r1, c1), (p2, r2, c2)


def transitive_closure(axioms):
    children, parents = map(set, zip(*axioms))

    roots = parents - children
    if not roots:
        raise ValueError("Taxonomy contains a cycle")

    c2p = {c: p for c, p in axioms}

    new_axioms = []
    for c in c2p:
        p = c
        visited = {c}
        while p in children:
            p = c2p[p]
            if p in visited:
                raise ValueError(f"Taxonomy contains a cycle involving {p}")
            visited.add(p)
            new_axioms.append((c, p))

    return set(new_axioms) | set(axioms)
