from .symbol import Rel, Sym
from typing import Union, Callable, Dict
from operator import and_, or_, invert, not_


class AxiomOp:
    """
    Represent a logical operator between one or several axioms.

    An operator indicates how to combine axioms in order to create new axioms. Essentially, it has an arity k and a
    combining function. The arity indicates the number of axiom to combine (1 for the negation, 2 for conjunction and
    disjunction), the combining function is the operator per se.

    Some operators are defined below: NEG, AND, OR, REM. New operators can be added or created on-the-fly.
    """
    def __init__(self, symbol_or_naming_func,
                 func: Callable,
                 arity: int,
                 bfunc: Union[None, Callable] = None,
                 sfunc: Union[None, Callable] = None,
                 params: Union[None, Dict] = None) -> None:
        """
        An operator defines 3 functions that indicate how to build the axiom
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

        By default, `func`=`bfunc`=`sfunc`, but you can provide custom functions for each.

        :param symbol_or_naming_func: indicates how to name the new axiom, based on the names of the components. If a
        string, it is assumed to be a logical symbol, such as ∧, ¬, ∨. If a callable, then it is assumed to be a
        functions taking as many strings as the arity as input, and returning a new string. This way, you can provide
        custom naming schemes for your operators.
        :param func: vector function.
        :param arity: int, arity of the operator. If -1, the arity is variable. See RemainderAximo for a use case.
        :param bfunc: boolean function. If None, `func` is used.
        :param sfunc: set function. If None, `func` is used.
        :param params: not used anymore.
        """
        self.symbol = symbol_or_naming_func if callable(symbol_or_naming_func) else str(symbol_or_naming_func)
        self.func = func
        self.bfunc = func if bfunc is None else bfunc
        self.sfunc = func if sfunc is None else sfunc
        self.arity = arity
        self.params = params if params is not None else dict()

OR = AxiomOp(Sym.OR, or_, 2)
AND = AxiomOp(Sym.AND, and_, 2)
NEG = AxiomOp(Sym.NEG, invert, 1, not_)

def remainder_vfunc(base_vec, *sub_vecs):
    return and_(base_vec, ~sum(sub_vecs))

def remainder_bfunc(base_axiom, *sub_axioms):
    return base_axiom and not any(sub_axioms)

def remainder_sfunc(base_items, *sub_items):
    for sub_item in sub_items:
        base_items -= sub_item
    return base_items

def remainder_naming_func(base_component, *sub_components):
    return f"*({base_component.name})"

REM = AxiomOp(remainder_naming_func, func=remainder_vfunc, arity=-1, bfunc=remainder_bfunc, sfunc=remainder_sfunc)