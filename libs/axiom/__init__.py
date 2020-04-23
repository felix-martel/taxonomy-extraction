# from .base import Axiom
# from .atomic import AtomicAxiom, TopAxiom, EmptyAxiom, Concept, Existential
# from .composed import NaryAxiom

from .base import Axiom, NaryAxiom
from .atomic import AtomicAxiom, TopAxiom, EmptyAxiom, Concept, Existential
from .operators import AND, OR, NEG, AxiomOp