import random
from typing import Set, Union, Tuple, Optional

from libs.graph import KnowledgeGraph
from libs.axiom import Axiom, NaryAxiom, TopAxiom, AtomicAxiom, Concept, Existential, NEG

InstanceIds = Set[int]
InstanceNames = Set[str]
Instances = Union[InstanceIds, InstanceNames]
Sampled = Tuple[Instances, int]

__GRAPH = None

def set_graph(graph: Optional[KnowledgeGraph]):
    global __GRAPH
    __GRAPH = graph


def __sample(instances: Set[int], size: int, force_size: bool) -> Sampled:
    n = len(instances)
    if n < size:
        if force_size:
            raise ValueError(f"Not enough items to sample from (expected at least {size}, got {n})")
        return instances, n
    return set(random.sample(instances, size)), n


def __check_graph(graph):
    if graph is None:
        raise ValueError("You must provide a graph, whether by using 'instance_sampling.set_graph' function or by"
                         "declaring parameter 'graph'")
    return


def sample_any(size, graph: Optional[KnowledgeGraph] = None, exclude_literals=True, force_size=False) -> Sampled:
    """
    Sample entities uniformly from the graph, with no condition on types or anything. If exclude_literals, then only
    typed entities will be sampled (ie all h such that (h, rdf:type, t) is in the graph, for some type t)
    """
    if graph is None:
        graph = __GRAPH
    __check_graph(graph)
    if exclude_literals:
        hs = graph._r[graph.rel.to_id("rdf:type")].keys()
    else:
        hs = graph.ent.idx.keys()
    return __sample(hs, size, force_size)


def sample_from(axiom: Axiom, size: int, graph: Optional[KnowledgeGraph] = None,
                exclude_literals: bool = True, force_size: bool = False) -> Sampled:
    if graph is None:
        graph = __GRAPH
    __check_graph(graph)
    if axiom is TopAxiom:
        return sample_any(size, graph, exclude_literals, force_size)
    else:
        instances = instances_from_axiom(axiom, graph)
        return __sample(instances, size, force_size)


def instances_from_atom(atom: AtomicAxiom, graph: Optional[KnowledgeGraph] = None) -> InstanceIds:
    """
    Get all entities verifying an atomic axiom
    """
    if graph is None:
        graph = __GRAPH
    __check_graph(graph)
    if atom is TopAxiom:
        # Any instance would work
        return graph.ent.idx.keys()
    elif isinstance(atom, (Existential, Concept)):
        r = atom.rel
        c = atom.concept
    else:
        raise NotImplementedError(f"Function 'instances_from_atom' is not defined for {atom}")
    isa = graph.rel.to_id("rdf:type")
    rid = graph.rel.to_id(r)
    cid = graph.ent.to_id(c.concept) if c is not TopAxiom else None
    if c.is_singleton or c is TopAxiom:
        instances = {h for h, _, _ in graph.find_triples(r=rid, t=cid)}
    else:
        instances = {h for h, _, t in graph.find_triples(r=rid) if (t, isa, cid) in graph}
    return instances


def instances_from_axiom(axiom: Axiom, graph: Optional[KnowledgeGraph] = None) -> InstanceIds:
    if graph is None:
        graph = __GRAPH
    __check_graph(graph)
    if isinstance(axiom, AtomicAxiom):
        return instances_from_atom(axiom, graph)
    elif isinstance(axiom, NaryAxiom):
        if axiom.op is NEG:
            # TODO What if we reimplemented neg axioms as binary axioms Top - Neg?
            raise NotImplementedError("'instances_from_axiom' cannot be called directly with negative axioms")
        return axiom.op.sfunc(*(instances_from_axiom(sub, graph) for sub in axiom.components))
    else:
        raise NotImplementedError(f"'instances_from_axiom' not supported for {axiom}")
