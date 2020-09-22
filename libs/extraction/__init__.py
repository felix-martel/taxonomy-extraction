from .global_max import extract_axioms as extract_global
from .local_max import extract_axioms as extract_local
from .probabilistic import extract_axioms as extract_proba
from .extractor import extract_axioms, TaxonomyExtractor


# Aliases
hard_mapping = extract_global
soft_mapping = extract_proba
