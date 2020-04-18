import numpy as np
from typing import Union, Iterable, Generator
from .operators import AND, OR, NEG
from .composed import NaryAxiom

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

    # TODO Implement scoring functions
    # TODO Implement sampling methods
    # TODO Add a 'Axiom.is_open' method
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



