from typing import TypeVar, List, Iterable, Union, Optional, Dict, Generic, Hashable, Type, Tuple, Iterator, Mapping, \
    no_type_check

A = TypeVar("A", bound=Hashable)
B = TypeVar("B", bound=Hashable)
AB = Union[A, B]

class Mapper(Generic[A, B], Iterable[Tuple[A, B]]):
    """
    Two-Way Mapping

    Usage example:
    ```
    >>> vocab = "abcdef"
    >>> m = Mapper(vocab, "letter", "id")
    >>> len(m)
    6
    >>> "a" in m.letters, 0 in m.letters
    True, False
    >>> m.to_id("a")
    0
    >>> m.to_letter(0)
    "a"
    >>> m.to_letter("a")
    "a"
    >>> print(*m.letters)
    a b c d e f
    >>> dict(m)
    {"a": 0, "b": 1, "c": 2, "d": 3}
    >>> dict(m.reverse())
    {0: "a", 1: "b", 2: "c", 3: "d"}
    ```
    """

    def __init__(self, data, class_a: str = "a", class_b: str = "b",
                 type_a: Optional[Type] = None, type_b: Optional[Type] = None, auto_id: bool = False) -> None:
        if auto_id:
            type_b = int
        self.a2b, self.b2a = self.process(data)
        self.autotype, self.type_a, self.type_b = self.check_types(type_a, type_b)
        self.__name_a, self.__name_b = "a", "b"
        self.name_a = class_a
        self.name_b = class_b
        self.auto_id = auto_id

    def add(self, a: A, b: Optional[B] = None, exist_ok: bool = True):
        if b is None:
            if not self.auto_id:
                raise ValueError("Since auto_id=False, you must provide a value for parameter b")
            else:
                b = max(self.b2a) + 1
        if a in self.a2b and not exist_ok:
            raise ValueError(f"'{a}' already in self.{self.name_a}s. Set exist_ok=True to insert anyway")
        if b in self.b2a and not exist_ok:
            raise ValueError(f"'{b}' already in self.{self.name_b}s. Set exist_ok=True to insert anyway")
        self.a2b[a] = b
        self.b2a[b] = a

    def __repr__(self):
        return f"Mapper(from={self.name_a}, to={self.name_b})"
    
    def __getitem__(self, item: AB) -> AB:
        if not self.autotype:
            raise TypeError(f"Can't access items with brackets when 'autotype=False'. Use 'self.to_{self.name_a}()' "
                            f"or 'self.to_{self.name_b}()' instead.")
        if isinstance(item, self.type_a):
            return self.to_b(item)
        elif isinstance(item, self.type_b):
            return self.to_a(item)
        else:
            raise TypeError(f"Type '{type(item)}' is not recognized. Expected {self.type_a} or {self.type_b}")
        

    @classmethod
    def from_iterable(cls, l: Iterable[A], name: str = "item", id_name: str = "id") -> "Mapper[A, int]":
        data = {a: b for b, a in enumerate(l)}
        return cls(data, class_a=name, class_b=id_name)

    @property
    def name_a(self) -> str:
        return self.__name_a

    @name_a.setter
    def name_a(self, name) -> None:
        if name != "a":
            name_a, name_b = name, self.name_b
            setattr(self, f"{name_a}_to_{name_b}", self.a_to_b)
            setattr(self, f"{name_b}_to_{name_a}", self.b_to_a)
            setattr(self, f"{name_a}s", self.iter_as)
            setattr(self, f"to_{name_a}", self.to_a)
            setattr(self, f"to_{name_a}s", self.to_as)
        self.__name_a = name

    @property
    def name_b(self) -> str:
        return self.__name_b

    @name_b.setter
    def name_b(self, name) -> None:
        if name != "b":
            name_a, name_b = self.name_a, name
            setattr(self, f"{name_a}_to_{name_b}", self.a_to_b)
            setattr(self, f"{name_b}_to_{name_a}", self.b_to_a)
            setattr(self, f"{name_b}s", self.iter_bs)
            setattr(self, f"to_{name_b}", self.to_b)
            setattr(self, f"to_{name_b}s", self.to_bs)
        self.__name_b = name

    @classmethod
    def __to_dict(cls, data: Union[Dict[A, B], Iterable[Tuple[A, B]]]) -> Dict[A, B]:
        return dict(data)

    @classmethod
    def process(cls, data) -> Tuple[Dict[A, B], Dict[B, A]]:
        a2b = cls.__to_dict(data)
        b2a = {v: k for k, v in a2b.items()}
        return a2b, b2a

    def check_types(self, type_a: Optional[Type] = None, type_b: Optional[Type] = None
                    ) -> Tuple[bool, Optional[Type], Optional[Type]]:
        if not self:
            return True, None, None
        if type_a is None:
            type_a = type(next(iter(self.iter_as)))
        if type_b is None:
            type_b = type(next(iter(self.iter_bs)))
        a_ok = all(isinstance(a, type_a) for a in self.iter_as)
        b_ok = all(isinstance(b, type_b) for b in self.iter_bs)
        if not a_ok:
            raise TypeError(f"Not all values in 'a' have type {type_a}")
        if not b_ok:
            raise TypeError(f"Not all values in 'b' have type {type_b}")
        distinct = type_a != type_b
        autotype = a_ok and b_ok and distinct
        return autotype, type_a, type_b

    def a_to_b(self, a: A) -> B:
        return self.a2b[a]

    def b_to_a(self, b: B) -> A:
        return self.b2a[b]

    @property
    def iter_as(self) -> Iterable[A]:
        return self.a2b.keys()

    @property
    def iter_bs(self) -> Iterable[B]:
        return self.b2a.keys()

    def to_b(self, item: AB, allow_autotype: bool = False) -> B:
        if allow_autotype and ((self.autotype and isinstance(item, self.type_b))
                               or (not self.autotype and item in self.a2b)):
            return item
        return self.a_to_b(item)

    def to_bs(self, items: Iterable[AB], allow_autotype: bool = False) -> List[B]:
        return [self.to_b(item, allow_autotype) for item in items]

    def to_as(self, items: Iterable[AB], allow_autotype: bool = False) -> List[A]:
        return [self.to_a(item, allow_autotype) for item in items]

    def to_a(self, item: AB, allow_autotype: bool = False) -> A:
        if allow_autotype and ((self.autotype and isinstance(item, self.type_a))
                or (not self.autotype and item in self.b2a)):
            return item
        return self.b_to_a(item)

    def __len__(self) -> int:
        return len(self.a2b)

    def __bool__(self) -> bool:
        return bool(len(self))

    def __contains__(self, item) -> bool:
        return item in self.a2b or item in self.b2a

    def __iter__(self) -> Iterator[Tuple[A, B]]:
        for a, b in self.a2b.items():
            yield a, b

    def reverse(self) -> "Mapper[B, A]":
        return Mapper(
            self.b2a,
            class_a=self.name_b,
            class_b=self.name_a,
            type_a=self.type_b,
            type_b=self.type_a
        )


if __name__ == "__main__":
    vocab = "abcdef"
    m = Mapper.from_iterable(vocab, "letter")

    print("m =", m)
    print("len(m) =", len(m))

    print("'a' in m.letters:", 'a' in m.letters, "/ 0 in m.letters:", "b" in m.letters)
    print("'a' ->", m.to_id("a"), "/ 0 ->", m.to_letter(0))
    print("letters:", *m.letters)
    print("to dict:", dict(m))
    print("rev dict:", dict(m.reverse()))
