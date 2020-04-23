from .symbol import Rel, Sym
from typing import Union, Callable, Dict
from operator import and_, or_, invert, not_

class AxiomOp:
    def __init__(self, symbol,
                 func: Callable,
                 arity: int,
                 bfunc: Union[None, Callable] = None,
                 sfunc: Union[None, Callable] = None,
                 params: Union[None, Dict] = None) -> None:
        self.symbol: str = str(symbol)
        self.func = func
        self.bfunc = func if bfunc is None else bfunc
        self.sfunc = func if sfunc is None else sfunc
        self.arity = arity
        self.params = params if params is not None else dict()

OR = AxiomOp(Sym.OR, or_, 2)
AND = AxiomOp(Sym.AND, and_, 2)
NEG = AxiomOp(Sym.NEG, invert, 1, not_)

