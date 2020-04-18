from libs.axiom import Axiom
from libs.axiom.atomic import AtomicAxiom
from libs.axiom.operators import AxiomOp


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
