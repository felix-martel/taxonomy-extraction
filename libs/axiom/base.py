from abc import ABC

from .symbol import Sym, Rel
import numpy as np
from typing import Union, Iterable, Generator
from .operators import AxiomOp, AND, OR, NEG
from libs.graph import KnowledgeGraph


class Axiom:
    """
    Base class for representing an axiom.

    An axiom has:
    - a name
    - a list of subaxioms (?)
    - a vector representation (optional), that is a boolean vector v such that v_i = True iff axiom holds for entity i
    - a check function, to verify if axiom holds for a given entity and a given knowledge graph
    - scoring functions, to evaluate its coverage, specificity and partition score over a given set of entities

    Axioms can be combined, using conjunctions, disjunctions and negations.
    An axiom is either:
    - an atomic axiom (such that C, ∃R.C, ∃R.{v}, ⊤...)
    - a combination of axioms, given by an operation `op` with arity `k`, and `k` axioms

    Special axioms include:
    - EmptyAxiom, an uninitialised axiom (used for the "axiom improvement" step)
    - TopAxiom, representing the axiom ⊤ (always True)
    - RemainderAxiom(ax), representing ax \ subaxes

    # TODO
    - Implement scoring functions
    - Implement sampling methods
      - Add a 'Axiom.is_open' method
    """
    def __init__(self, name: str,
                 vec: Union[None, np.ndarray] = None,
                 components: Union[None, Iterable] = None) -> None:
        self._name = name
        self._vec = vec
        self.components = components if components is not None else []
        self.is_atomic = not self.components
        self._has_vec: Union[None, bool] = vec is None

    @property
    def has_vec(self) -> bool:
        return self.vec is not None

    def __str__(self):
        return str(self.name)

    def __repr__(self):
        return f"{type(self).__name__}({self})"

    @property
    def vec(self):
        return self._vec

    @vec.setter
    def vec(self, value):
        self._vec = value

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, n: str) -> None:
        self._name = n

    @property
    def dim(self) -> Union[None, int]:
        try:
            return len(self.vec)
        except AttributeError:
            return None

    def clear_memory(self) -> None:
        self.vec = None

    def atoms(self) -> Generator["Axiom", None, None]:
        if self.is_atomic:
            yield self
        elif self.components:
            for component in self.components:
                yield from component.atoms()

    def __and__(self, other: Union["Axiom", np.ndarray]) -> Union["Axiom", np.ndarray]:
        if isinstance(other, np.ndarray):
            # Here we assume 'self.vec' is not None (Note to self: is there a use case for this?)
            return self.vec & other
        return NaryAxiom(AND, self, other)

    def __or__(self, other: Union["Axiom", np.ndarray]) -> Union["Axiom", np.ndarray]:
        if isinstance(other, np.ndarray):
            # Here we assume 'self.vec' is not None
            return self.vec | other
        return NaryAxiom(OR, self, other)

    def __invert__(self) -> "Axiom":
        if isinstance(self, NaryAxiom) and self.op is NEG:
            return self.components[0]
        return NaryAxiom(NEG, self)


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
        self.concept = concept if concept is not None else singleton
        self.is_singleton = concept is None
        self.is_top = False
        self.A = None
        self.i = None

    def set_vec(self, A: np.ndarray, i: int) -> None:
        self.A = A
        self.i = i

    @property
    def vec(self) -> Union[None, np.ndarray]:
        if self.A is None:
            return None
        return self.A[:self.i]

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
    def __init__(self, rel: str, concept: Union[None, Concept], vec=None) -> None:
        if concept is None:
            concept = TopAxiom
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


class NaryAxiom(Axiom):
    """
    Represent a N-ary axiom, that is the combination of N axioms by a N-ary operator.
    A `NaryAxiom` is defined by an operator `op` with arity $N$ (*e.g* NEG with arity 1, AND, OR with arity 2), and
    $N$ axioms $\alpha_1, \alpha_2, \ldots \alpha_N$. Then, the NaryAxiom is simply:
    $$\alpha_\text{n-ary} = op(\alpha_1, \ldots, \alpha_k)$$
    """
    def __init__(self, op: AxiomOp, *axioms: Axiom):
        axioms = list(axioms)
        if len(axioms) != op.arity:
            raise ValueError(f"Expected {op.arity} axioms, found {len(axioms)}")
        super().__init__("", None, components=axioms)
        self.op = op
        self._has_vec = None

    @property
    def has_vec(self) -> bool:
        if self._has_vec is None:
            self._has_vec = all(ax.has_vec for ax in self.components)
        return self._has_vec

    @property
    def name(self):
        if self.op.arity == 1:
            component = self.components[0]
            name = component.name
            if not isinstance(component, AtomicAxiom) and component.op.arity > 0:
                name = f"({name})"
            return f"{self.op.symbol}{name}"
        names = []
        for ax in self.components:
            name = ax.name
            if not isinstance(ax, AtomicAxiom) and ax.op.arity > 1 and ax.op is not self.op:
                name = f"({name})"
            names.append(name)
        return self.op.symbol.join(names)

    @property
    def vec(self):
        if self.has_vec:
            return self.op.func(*(ax.vec for ax in self.components))
        return None

    def holds_for(self, entity, graph, **params):
        """
        Check if axiom `self` is verified by `entity` in the knowledge graph `graph`
        """
        return self.op.bfunc(*(ax.holds_for(entity, graph, **params) for ax in self.components))
