import numpy as np
from typing import Union, Iterable, Generator

from libs.utils.misc import void
from .operators import AND, OR, NEG, REM, AxiomOp
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
    - a combination of axioms, given by an operation `op` with arity `k`, and `k` axioms. -1 indicates a varying arity

    Special axioms include:
    - EmptyAxiom, an uninitialised axiom (used for the "axiom improvement" step)
    - TopAxiom, representing the axiom ⊤ (always True)
    - RemainderAxiom(ax), representing ax \ subaxes, see the corresponding class

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

    def serialize(self):
        # TODO: add a serialization/deserialization method (axiom to string)
        pass

    @classmethod
    def deserialize(cls, data):
        # TODO: deserialization (string to axiom)
        pass

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
            raise ValueError(f"Unrecognized score function '{how}'. Valid values "
                             "are 'harmonic', 'arithmetic', 'prod', 'xor'")

        return cov, spe, sco


class NaryAxiom(Axiom):
    """
    Represent a N-ary axiom, that is the combination of N axioms by a N-ary operator.

    The N axioms are stored into the `components` array, and combined using an operator `op`. Some operators are defined
    in `libs.axiom`, such as NEG (negation), OR (disjunction), AND (conjunction), REM (remainder, see `RemainderAxiom`).

    An operator `op` from class `libs.axiom.operators.AxiomOp` defines 3 functions that indicate how to build the axiom
    from its components: `func`, `bfunc` ('b' for 'boolean'), `sfunc` ('s' for 'set'). Let A be a k-ary axiom, and
    B1, ..., Bk its components (each Bi is instance of class `Axiom`). Let e be an entity from the graph. We write A(e)
    if axiom A holds for the entity e, E(A) for the set of entities that verify A, and V(A) the boolean vector such
    that V(A)_i = 1 if A holds for e_i, 0 otherwise. Then:

    - `bfunc` indicates how to compute A(e) from B1(e), ..., Bk(e) :
    A(e) = bfunc(B1(e), B2(e), ..., Bk(e))
    It is used in method `holds_for`

    - `sfunc` indicates how to compute E(A) from E(B1), ..., E(Bk) :
    E(A) = sfunc(E(B1), E(B2), ..., E(Bk))
    It is used for sampling items, e.g in the `libs.sampling.GraphSampler` class

    - `func` indicates how to compute V(A) from V(B1), ..., V(Bk) :
    V(A) = sfunc(V(B1), V(B2), ..., V(Bk))
    It is used in property `vec`

    Example: for the operator AND, `bfunc` is the standard AND operator (lambda x,y: x and y), `func` is the elementwise
    AND operator (numpy.logical_and), and `sfunc` is the set intersection (lambda A, B: A & B).

    New operators can be added or created on-the-fly. By default, `func`=`bfunc`=`sfunc`, but you can provide custom
    functions for each, as well as a custom naming scheme.
    """
    def __init__(self, op: AxiomOp, *axioms: Axiom):
        axioms = list(axioms)
        if len(axioms) != op.arity and op.arity != -1:
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
        if callable(self.op.symbol):
            return self.op.symbol(*self.components)
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

    A RemainderAxiom is a special axiom *a meaning: "instances from 'a' that don't belong to known subclasses of 'a'".
    See Section 5.2.2 and especially Equation 5.3 of my thesis for details. A RemainderAxiom is thus composed of
    a base axiom A and k sub-axioms B1, B2, ... Bk (representing the 'known subclasses' of A). Its logical meaning
    can be written as:
    A and not (B1 or B2 or ... or Bk)

    The sub-axioms B1, ..., Bk can change over time, and the RemainderAxiom cannot know the current known classes. Thus,
    they must be added manually before any sampling takes place. Two methods are provided: `RemainderAxiom.add` (for
    adding one subaxiom) and `RemainderAxiom.update` (for adding several subaxioms).

    Entities from a RemainderAxiom can be properly sampled once all subaxioms have been added.
    """
    def __init__(self, base : Axiom, *subaxioms : Axiom):
        #rem = AxiomOp("*", void, 1)
        super().__init__(REM, base, *subaxioms)

    @property
    def base(self):
        return self.components[0]

    def add(self, subaxiom):
        """Add a new subaxiom B to the RemainderAxiom"""
        self.components.append(subaxiom)

    def update(self, subaxioms):
        """Add multiple subaxioms B1, ..., Bk to the RemainderAxiom"""
        self.components.extend(subaxioms)

    def set_subaxioms(self, subaxioms):
        """Set subaxioms B1, ..., Bk (removing existing subaxioms"""
        self.components = [self.base, *subaxioms]


