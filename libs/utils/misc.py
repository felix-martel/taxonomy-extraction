from functools import singledispatch
from typing import Callable, Dict, TypeVar


T = TypeVar("T")


def safe_divide(p, q):
    if q == 0: return 0.0
    return p / q


@singledispatch
def namer(p) -> Callable[[T], str]:
    """
    Return a naming function.

    The default naming function is 'str', but you can also use an attribute name,
    a callable, a dict.

    Example:
    ```
    >>> x = 2
    >>> get_name2 = namer("__class__")
    >>> get_name3 = namer("{:03d}".format)
    >>> get_name4 = namer({1: "one", 2: "two", 3: "three"}
    >>> get_name2(x), get_name3(x), get_name4(x)
    ('int', '002', 'two')
    ```
    """
    return str


@namer.register(str)
def _(p: str) -> Callable[[T], str]:
    def get_name(item):
        return str(getattr(item, p))
    return get_name


@namer.register(Callable)
def _(p: Callable[[T], str]):
    return p


@namer.register(dict)
def _(p: Dict[T, str]):
    return p.get
