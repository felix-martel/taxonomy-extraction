from typing import Union
from abc import ABC

import numpy as np

from .symbol import Rel, Sym
from .base import Axiom
from ..graph import KnowledgeGraph


class AtomicAxiom(Axiom, ABC):
    """
    Abstract class for atomic axioms.

    Atomic axioms (*atoms* in short) are the basic axiom building blocks: concepts (such as `dbo:Person` or
    `dbo:Place`), existential restrictions ∃R.C, value properties ∃R.{v}, datatype properties, or special axioms like ⊤.
    """
    def __init__(self, name: str, vec: Union[None, np.ndarray] = None) -> None:
        super().__init__(name, vec)

    def holds_for(self, entity: int, graph: KnowledgeGraph, **params) -> bool:
        pass


class Concept(AtomicAxiom):
    """
    Represent a Concept.

    A concept is the simplest axiom pattern. A concept `C` holds for an entity `x` if the triple `(x, rdf:type, C)`
    exists in the knowledge graph. We also add a special `Concept` element, named `TopAxiom`, for axiom ⊤.
    Another special case of concept is *singleton concept* {v}. Singleton concepts contain only one elements, and
    they're mostly useful to represent value properties ∃R.{v}.
    """
    rel = str(Rel.IS_A)

    def __init__(self, concept=None, singleton=None):
        if concept is None and singleton is None:
            raise ValueError("You must specify at least one of 'concept' or 'singleton'")
        else:
            name = concept if concept is not None else "{" + singleton + "}"
        super().__init__(name)
        self.rel = "rdf:type"
        self.concept = concept if concept is not None else singleton
        self.is_singleton = concept is None
        self.is_top = False
        self.A = None
        self.i = None

    # def set_vec(self, A: np.ndarray, i: int) -> None:
    #     self.A = A
    #     self.i = i
#
    # @property
    # def vec(self) -> Union[None, np.ndarray]:
    #     if self.A is None:
    #         return None
    #     return self.A[:, self.i]

    def clear_memory(self) -> None:
        self.A = None
        self.i = None

    def holds_for(self, entity: int, graph: KnowledgeGraph, **params) -> bool:
        if self.is_top:
            return True
        # If 'self.concept' is a singleton, then ∃v, self = {v}, so e∈{v} <=> e == v
        if self.is_singleton:
            return graph.ent.to_name(entity) == self.concept
        # Else, self=C with C a class, and we just have to check if (e, rdf:type, C) is in the graph
        r, c = graph.rel.to_id(self.rel), graph.ent.to_id(self.concept)
        return (entity, r, c) in graph


TopAxiom = Concept(str(Sym.TOP))
TopAxiom.is_top = True


class Existential(AtomicAxiom):
    """
    Represent an existential restriction.

    An existential restriction  `∃R.C` is defined by a relation `R` and a concept `C`. It holds for entity `x` if
    `x` is linked to another entity `y` by the relation `R`, with `y` belonging to concept `C`:
    $$x \in \exists R.C \iff \exists y \in \mathcal{E}, (x, R, y) \in \Delta \land C(y)$$
    """
    def __init__(self, rel: str, concept: Union[None, str, Concept], vec=None) -> None:
        if concept is None:
            concept = TopAxiom
        elif isinstance(concept, str):
            concept = Concept(concept)
        name = f"{Sym.EXISTS}{rel}.{concept.name}"
        super().__init__(name, vec)
        self.rel = rel
        self.concept = concept

    def holds_for(self, entity: int, graph: KnowledgeGraph, **params) -> bool:
        # self = ∃r.C, so self holds for entity e iff ∃y in the graph, such that (e, r, y) and C(y)
        r = graph.rel.to_id(self.rel)
        return any(self.concept.holds_for(y, graph) for _, _, y in graph.find_triples(h=entity, r=r))


class EmptyAxiom(Axiom):
    """
    Represent an unitialized axiom. Used for the axiom improvement algorithm, when the list of candidate axioms
    hasn't been initialized.
    """
    def __init__(self):
        super().__init__("__empty__", vec=None)

    def __and__(self, other): return other

    def __or__(self, other): return other

    def __rand__(self, other): return other

    def __ror__(self, other): return other

    def __invert__(self): return self

    def coverage(self, mask=None): return 0.0

    def specificity(self, mask=None): return 0.0

    def xor_score(self, mask): return 0.0

    def evaluate(self, mask, how="arithmetic"):
        return 0, 0, 0
