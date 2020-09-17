# from .base import Axiom
# from .atomic import AtomicAxiom, TopAxiom, EmptyAxiom, Concept, Existential
# from .composed import NaryAxiom

from .base import Axiom, NaryAxiom, RemainderAxiom
from .atomic import AtomicAxiom, TopAxiom, EmptyAxiom, Concept, Existential
from .operators import AND, OR, NEG, REM, AxiomOp


def is_atomic(axiom):
    return axiom.is_atomic

def is_concept(axiom):
    return isinstance(axiom, Concept)

def is_existential(axiom):
    return isinstance(axiom, Existential)

def is_empty(axiom):
    return isinstance(axiom, EmptyAxiom)

def is_composed(axiom):
    return isinstance(axiom, NaryAxiom)

def is_top(axiom):
    return axiom is TopAxiom

def is_remainder(axiom):
    return isinstance(axiom, RemainderAxiom)

def is_neg(axiom):
    return is_composed(axiom) and axiom.op == NEG

def is_pos(axiom):
    return not is_neg(axiom)


