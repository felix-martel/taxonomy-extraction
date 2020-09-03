import random
from typing import Set, Union, Tuple, Optional

from libs.graph import KnowledgeGraph
from libs.axiom import Axiom, NaryAxiom, TopAxiom, AtomicAxiom, Concept, Existential, NEG

InstanceIds = Set[int]
InstanceNames = Set[str]
Instances = Union[InstanceIds, InstanceNames]
Sampled = Tuple[Instances, int]


def _sample(instances: Set[int], size: int, force_size: bool) -> Sampled:
    n = len(instances)
    if n < size:
        if force_size:
            raise ValueError(f"Not enough items to sample from (expected at least {size}, got {n})")
        return instances, n
    return set(random.sample(instances, size)), n

class GraphSampler(object):
    """
    Object for sampling instances within a graph using axioms
    """

    def __init__(self, graph, restricted_ids=None):
        self.graph = graph
        self.valid_ids = restricted_ids

    def _sample(self, instances: Set[int], size: int, force_size: bool) -> Sampled:
        if self.valid_ids is not None:
            instances &= self.valid_ids
        n = len(instances)
        if n < size:
            if force_size:
                raise ValueError(f"Not enough items to sample from (expected at least {size}, got {n})")
            return instances, n
        return set(random.sample(instances, size)), n

    def any(self, size, exclude_literals=True, force_size=False) -> Sampled:
        """
            Sample entities uniformly from the graph, with no condition on types or anything. If exclude_literals, then only
            typed entities will be sampled (ie all h such that (h, rdf:type, t) is in the graph, for some type t)
            """
        if exclude_literals:
            hs = self.graph._r[self.graph.rel.to_id("rdf:type")].keys()
        else:
            hs = self.graph.ent.idx.keys()
        return self._sample(hs, size, force_size)

    def instances_from_atom(self, atom: AtomicAxiom, graph: Optional[KnowledgeGraph] = None) -> InstanceIds:
        """
        Get all entities verifying an atomic axiom
        """
        if atom is TopAxiom:
            # Any instance would work
            return self.graph.ent.idx.keys()
        elif isinstance(atom, Existential):
            r = atom.rel
            c = atom.concept
        elif isinstance(atom, Concept):
            r = atom.rel
            c = atom
        else:
            raise NotImplementedError(f"Function 'instances_from_atom' is not defined for {atom}")
        isa = self.graph.rel.to_id("rdf:type")
        rid = self.graph.rel.to_id(r)
        cid = self.graph.ent.to_id(c.concept) if c is not TopAxiom else None
        if c.is_singleton or c is TopAxiom:
            instances = {h for h, _, _ in self.graph.find_triples(r=rid, t=cid)}
        else:
            instances = {h for h, _, t in self.graph.find_triples(r=rid) if (t, isa, cid) in graph}
        return instances

    def instances_from_axiom(self, axiom: Axiom) -> InstanceIds:
        if isinstance(axiom, AtomicAxiom):
            return self.instances_from_atom(axiom, self.graph)
        elif isinstance(axiom, NaryAxiom):
            if axiom.op is NEG:
                # TODO What if we reimplemented neg axioms as binary axioms Top - Neg?
                raise NotImplementedError("'instances_from_axiom' cannot be called directly with negative axioms")
            return axiom.op.sfunc(*(self.instances_from_axiom(sub) for sub in axiom.components))
        else:
            raise NotImplementedError(f"'instances_from_axiom' not supported for {axiom}")

    def sample(self, axiom: Axiom, size: int, exclude_literals: bool = True, force_size: bool = False) -> Sampled:
        if axiom is TopAxiom:
            return self.any(size, exclude_literals, force_size)
        else:
            instances = self.instances_from_axiom(axiom)
            return self._sample(instances, size, force_size)


class NaiveGraphSampler(GraphSampler):
    """
    The NaiveGraphSampler finds items verifying a given axiom by checking, for all entities, whether it verifies the
    axiom. Hence the 'naive'. Useful for dev purposes on toy dataset. NOT FOR PRODUCTION (obviously)
    """
    def __init__(self, graph, restricted_ids=None):
        super().__init__(graph, restricted_ids)

    @property
    def ids(self):
        """Return an iterator over the list of valid ids"""
        return iter(self.valid_ids if self.valid_ids is not None else self.graph.ent.idx.keys())

    def sample(self, axiom: Axiom, size: int, exclude_literals: bool = True, force_size: bool = False):
        def is_valid(x):
            if axiom.holds_for(x, self.graph):
                if not exclude_literals:
                    return True
                else:
                    return x in self.graph._r[self.graph.isaid]
            return False
        instances = {x for x in self.ids if is_valid(x)}
        return self._sample(instances, size, force_size)
