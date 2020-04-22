import random

from libs.graph import KnowledgeGraph
from libs.axiom import TopAxiom, Axiom, AtomicAxiom, Concept, Existential


def sample_any(size, graph, exclude_literals=True, force_size=False):
    """
    Sample entities uniformly from the graph, with no condition on types or anything. If exclude_literals, then only
    typed entities will be sampled (ie all h such that (h, rdf:type, t) is in the graph, for some type t)
    """
    if exclude_literals:
        hs = graph._r[graph.rel.to_id("rdf:type")]
    else:
        hs = graph.ent.idx
    hs = list(hs)
    n = len(hs)
    if n < size:
        if force_size:
            raise ValueError(f"Not enough item to sample from (expected at least {size}, got {n}")
        return hs, n
    return random.sample(hs, size), n

def instances_from_atom(atom, graph):
    """
    Get all entities verifying an atomic axiom
    """
    if isinstance(atom, Existential):
        r = atom.rel
        c = atom.concept
    elif isinstance(atom, Concept):
        r = "rdf:type"
        c = atom
    else:
        raise NotImplementedError(f"Function 'instances_from_atom' is not defined for {atom}")




def instances_from_axiom(axiom, graph):
    pass