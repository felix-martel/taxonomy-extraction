import numpy as np
from typing import Union, Iterable, Generator

from libs.utils.misc import void
from .operators import AND, OR, NEG, AxiomOp
#  import libs.axiom.composed as composed
#  from .composed import NaryAxiom

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


    def __eq__(self, other):
        return self.name == other.name

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

    def holds_for(self, entity: int, graph, **params) -> bool:
        raise NotImplementedError("Method 'holds_for' is not defined for generic axioms.")

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

    def __hash__(self):
        return hash(self.name)

    def evaluate(self, mask, how="arithmetic"):
        n = len(mask)
        m = mask.sum()
        cov = np.sum(mask & self.vec) / m
        spe = 1 - np.sum(~mask & self.vec) / (n - m)

        if how == "harmonic":
            sco = 2 / (1/cov + 1/spe) if cov > 0 and spe > 0 else 0
        elif how == "arithmetic":
            sco = (cov + spe ) / 2
        elif how == "xor":
            sco = (m * cov + (n-m) * spe) / (n + m)
        elif how == "prod":
            sco = cov * spe
        else:
            raise ValueError("Unrecognized score function '{how}'. Valid values "
                             "are 'harmonic', 'arithmetic', 'prod', 'xor'")

        return cov, spe, sco


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
            if not component.is_atomic and component.op.arity > 0:
                name = f"({name})"
            return f"{self.op.symbol}{name}"
        names = []
        for ax in self.components:
            name = ax.name
            if not ax.is_atomic and ax.op.arity > 1 and ax.op is not self.op:
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


class RemainderAxiom(NaryAxiom):
    """
    Represent a RemainderAxiom for a taxonomic tree (maybe this should move to expressive/extractor ?)

    TODO: implement a proper `holds_for` method
    """
    def __init__(self, base : Axiom):
        rem = AxiomOp("*", void, 1)
        super().__init__(rem, base)

    @property
    def base(self):
        return self.components[0]



