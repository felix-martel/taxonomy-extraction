from .global_max import extract_axioms as extract_global
from .local_max import extract_axioms as extract_local
from .probabilistic import extract_axioms as extract_proba

methods = {
    "global": extract_global,
    "local": extract_local,
    "proba": extract_proba
}

def extract_axioms(F, method, *args, **kwargs):
    if method not in methods:
        raise TypeError(f"Unrecognized method: `{method}`. Valid values are `{'`, `'.join(methods)}`")
    func = methods[method]
    return func(F, *args, **kwargs)