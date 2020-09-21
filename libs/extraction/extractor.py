from .global_max import extract_axioms as extract_global
from .local_max import extract_axioms as extract_local
from .probabilistic import extract_axioms as extract_proba

from ..dataset import Dataset
from ..embeddings import load as load_embeddings
from ..cluster import clusterize

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


class TaxonomyExtractor(object):
    def __init__(self, dataset, embeddings=None, method="soft", clustering_params=None, mapping_params=None):
        if isinstance(dataset, Dataset):
            self.data = dataset
        else:
            self.data = Dataset.load(dataset)
        self.embeddings = load_embeddings(embeddings)
        self.params = dict(
            clustering=clustering_params if clustering_params is not None else dict(),
            mapping=mapping_params if mapping_params is not None else dict(),
            method=method
        )

        self.clu = None
        self.predicted = None

    @property
    def score_matrix(self):
        if self.clu is None:
            return None
        return self.clu.F()

    def run(self):
        self.clu = clusterize(self.data, self.embeddings, **self.params["clustering"])
        F = self.score_matrix()
        self.predicted = extract_axioms(F, self.params["method"], **self.params["mapping"])

        return self.predicted
