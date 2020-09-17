from .symbol import Rel, Sym
from typing import Union, Callable, Dict
from operator import and_, or_, invert, not_


class AxiomOp:
    def __init__(self, symbol_or_naming_func,
                 func: Callable,
                 arity: int,
                 bfunc: Union[None, Callable] = None,
                 sfunc: Union[None, Callable] = None,
                 params: Union[None, Dict] = None) -> None:
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