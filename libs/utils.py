import numpy as np
from libs.table import display_table
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# sphinx_gallery_thumbnail_number = 2

import math

def jitter(arr, amount=0.01):
    stdev = amount*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev

def millify(n):
    prefixes = ['','K','M','B','T']
    i = max(0,min(len(prefixes)-1,int(math.floor(0 if n == 0 else math.log10(abs(n))/3))))
    prec = 1 if i > 1 else 0
    return '{:.{prec}f}{}'.format(n / 10**(3 * i), prefixes[i], prec=prec)

def euclidean(a, b):
    return np.linalg.norm(a-b)

def get_memory_usage():
    with open('/proc/meminfo') as file:
        a = {l.split()[0][:-1]: int(l.split()[1]) for l in file}
        used = a["MemTotal"] - a["MemFree"] - a["Buffers"] - a["Cached"] - a["Slab"]
    return used

def check_memory_limit(max_size, verbose=True):    
    gig = 1024**2
    used = get_memory_usage()
    ok = used < max_size
    if verbose:
        status = "OK" if ok else "NOT OK"
        gig = 1024**2
        print(f"Memory usage {status}: {used/gig:.1f}G / {max_size/gig:.1f}G")
    return ok

def heatmap(datatable, x_labels, y_labels, title=None, prec=2, figsize=None):
    m, n = datatable.shape #len(y_labels), len(x_labels)

    if figsize is None: figsize = (3+0.5*m, 3+0.5*n)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(datatable)

    # We want to show all ticks...
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(m))
    # ... and label them with the respective list entries
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            text = ax.text(j, i, f"{datatable[i, j]:.{prec}f}",
                           ha="center", va="center", color="w")

    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    plt.show()

class Mapper:
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

    def __init__(self, data, class_a="a", class_b="b", type_a=None, type_b=None):
        self.a2b, self.b2a = self.process(data)
        self.autotype, self.type_a, self.type_b = self.check_types(type_a, type_b)
        self.__name_a, self.__name_b = "a", "b"
        self.name_a = class_a
        self.name_b = class_b

    @property
    def name_a(self):
        return self.__name_a

    @name_a.setter
    def name_a(self, name):
        if name != "a":
            name_a, name_b = name, self.name_b
            setattr(self, f"{name_a}_to_{name_b}", self.a_to_b)
            setattr(self, f"{name_b}_to_{name_a}", self.b_to_a)
            setattr(self, f"{name_a}s", self.iter_as)
            setattr(self, f"to_{name_a}", self.to_a)
            setattr(self, f"to_{name_a}s", self.to_as)
        self.__name_a = name

    @property
    def name_b(self):
        return self.__name_b

    @name_b.setter
    def name_b(self, name):
        if name != "b":
            name_a, name_b = self.name_a, name
            setattr(self, f"{name_a}_to_{name_b}", self.a_to_b)
            setattr(self, f"{name_b}_to_{name_a}", self.b_to_a)
            setattr(self, f"{name_b}s", self.iter_bs)
            setattr(self, f"to_{name_b}", self.to_b)
            setattr(self, f"to_{name_b}s", self.to_bs)
        self.__name_b = name
        
    def __to_dict(self, data):
        try:
            return dict(data)
        except ValueError:
            return {a: i for i, a in enumerate(data)}

    @classmethod
    def __to_dict(cls, data):
        try:
            return dict(data)
        except ValueError:
            return {a: i for i, a in enumerate(data)}

    @classmethod
    def process(cls, data):
        a2b = cls.__to_dict(data)
        b2a = {v: k for k, v in a2b.items()}
        return a2b, b2a

    def check_types(self, type_a=None, type_b=None):
        if not self:
            return True, None, None
        if type_a is None:
            type_a = type(next(iter(self.iter_as)))
        if type_b is None:
            type_b = type(next(iter(self.iter_bs)))
        a_ok = all(isinstance(a, type_a) for a in self.iter_as)
        b_ok = all(isinstance(b, type_b) for b in self.iter_bs)
        distinct = type_a != type_b
        autotype = a_ok and b_ok and distinct
        return autotype, type_a, type_b

    def a_to_b(self, a):
        return self.a2b[a]

    def b_to_a(self, b):
        return self.b2a[b]

    @property
    def iter_as(self):
        return self.a2b.keys()

    @property
    def iter_bs(self):
        return self.b2a.keys()

    def to_b(self, item):
        if (self.autotype and isinstance(item, self.type_b)) or (not self.autotype and item in self.a2b):
            return item
        return self.a_to_b(item)
    
    def to_bs(self, items):
        return [self.to_b(item) for item in items]
    
    def to_as(self, items):
        return [self.to_a(item) for item in items]

    def to_a(self, item):
        if (self.autotype and isinstance(item, self.type_a)) or (not self.autotype and item in self.b2a):
            return item
        return self.b_to_a(item)

    def __len__(self):
        return len(self.a2b)
    
    def __bool__(self):
        return bool(len(self))

    def __contains__(self, item):
        return item in self.a2b or item in self.b2a

    def __iter__(self):
        for a, b in self.a2b.items():
            yield a, b

    def reverse(self):
        return Mapper(
            self.b2a,
            class_a=self.name_b,
            class_b=self.name_a,
            type_a=self.type_b,
            type_b=self.type_a
        )