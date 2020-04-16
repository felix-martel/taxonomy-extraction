from collections import Counter
from enum import Enum
import numpy as np

class Rel(Enum):
    IS_A = "rdf:type"
    ROOT = "owl:Thing"


class Sym(Enum):
    NEG = "¬"
    EXISTS = "∃"
    TOP = "⊤"
    OR = "∨"
    AND = "∧"
    CUP = "⊔"
    CAP = "⊓"

def shorten_dbpedia_uri(name):
    return name.replace("dbo:", "")

class Axiom:
    """
    simple concept (is_a, C)
    rel (∃ r, C)
    neg
    and / or / ...
    """
    unary_ops = {
        Sym.NEG: "__invert__"
    }
    binary_ops = {
        Sym.OR: "__or__",
        Sym.AND: "__and__"
    }

    def __init__(self, name, vec, is_atomic=True, struc={}):
        self.name = name
        self._vec = vec
        self.is_atomic = is_atomic
        self.structure = struc
        
    @property
    def vec(self):
        return self._vec

    @property
    def dim(self):
        return len(self.vec)
    
    @property
    def full_name(self):
        if self.is_atomic:
            return self.name
        return f"({self.name})"
    
    def clear_memory(self):
        self._vec = None
    
    def coverage(self, mask=None):
        if mask is None:
            mask = np.ones(self.dim, dtype=bool)
        return np.sum(mask & self.vec) / mask.sum()
    
    def specificity(self, mask=None):
        if mask is None:
            mask = np.zeros(self.dim, dtype=bool)
        return 1 - self.coverage(~mask)
    
    def xor_score(self, mask):
        return np.mean(~mask ^ self.vec)
    
    def _prod_score(self, cov, spe):
        return cov * spe
    
    def prod_score(self, mask):
        return self._prod_score(self.coverage(mask), self.specificity(mask))
    
    def _harmonic_score(self, cov, spe):
        if not cov or not spe:
            return 0.0
        return 2 / (1/cov + 1/spe)    
    
    def harmonic_score(self, mask):
        return self._harmonic_score(self.coverage(mask), self.specificity(mask))
    
    def score(self, mask, metric="xor"):
        cov, spe = self.coverage(mask), self.specificity(mask)
        if metric == "xor":
            sco = self.xor_score(mask)
        elif metric == "prod":
            sco = self._prod_score(cov, spe)
        elif metric == "harmonic":
            sco = self._harmonic_score(cov, spe)
        else:
            raise ValueError(f"Unrecognized metric '{metric}'. Valid values are 'xor', 'prod', 'harmonic'.")
        return cov, spe, sco
    
    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return f"Axiom({self.name})"
    
    def __str__(self):
        return self.name
    
    def __eq__(self, other):
        # I have a bug here, can't tell why, so here's a dirty fix
        # return isinstance(other, Axiom) and other.name == self.name
        try:
            return other.name == self.name
        except AttributeError:
            return False

    def __and__(self, other):
        func = getattr(self.vec, "__and__") if self.vec is not None else lambda x: None
        if isinstance(other, np.ndarray): # not isinstance(other, Axiom):
            return func(other)
        name = self.__merge_name_and(other)
        if self.vec is not None and other.vec is not None and self.vec.shape == other.vec.shape:
            vec = func(other.vec)
        else:
            vec = None
        ax = Axiom(name, vec, False, {"arity": 2, "op": "and", "content": [self.structure, other.structure]})
        return ax

    def __or__(self, other):
        func = getattr(self.vec, "__or__") if self.vec is not None else lambda x: None
        if isinstance(other, np.ndarray): # not isinstance(other, Axiom):
            return func(other)
        name = self.__merge_name_or(other)
        if self.vec is not None and other.vec is not None and self.vec.shape == other.vec.shape:
            vec = func(other.vec)
        else:
            vec = None
        ax = Axiom(name, vec, False, {"arity": 2, "op": "or", "content": [self.structure, other.structure]})
        return ax
    
    def __merge_name(self, other, symbol):
        return f"{self.full_name}{symbol}{other.full_name}"
    
    def __merge_name_or(self, other):
        return self.__merge_name(other, Sym.OR.value)
    
    def __merge_name_and(self, other):
        return self.__merge_name(other, Sym.AND.value)

    def __invert__(self):
        name = f"{Sym.NEG.value}{self.full_name}"
        vec = getattr(self.vec, "__invert__")() if self.vec is not None else None
        ax = Axiom(name, vec, False, {"arity": 1, "op": "neg", "content": [self.structure]})
        return ax
    
class RemainderAxiom(Axiom):
    def __init__(self, axiom):
        if isinstance(axiom, RemainderAxiom) or hasattr(axiom, "base") or axiom.structure["op"] == "rem":
            self = axiom
        else:
            struc = dict(
                arity=1,
                op="rem",
                content=[axiom.structure]
            )
            name = f"{axiom}/..."
            super().__init__(name, None, is_atomic=False, struc=struc)
            self.base = axiom

class TopAxiom(Axiom):
    def __init__(self, dim=None):
        vec = np.ones((dim, ), dtype=bool) if dim is not None else None
        name = "__top__"
        struc = dict(arity=0, op="__top__", content=[("rdf:type", "owl:Thing")])
        super().__init__(name, vec, is_atomic=True, struc=struc)
        
class AtomicAxiom(Axiom):
    def __init__(self, a, b, vec=None, idx=None, A=None):
        struc = {
            "arity": 0,
            "op": None,
            "content": [(a, b)]
        }
        #a = a.replace("dbo:", "")
        #b = b.replace("dbo:", "")
        is_concept = False
        is_existential = False
        
        if b == Rel.ROOT.value:
            b = Sym.TOP.value
        if a == Rel.IS_A.value:
            name = b
            is_concept = True
        else:
            name = f"{Sym.EXISTS.value}{a}.{b}"
            is_existential = True
        self.idx = idx
        self.A = A
        self.rel = a
        self.obj = b
        super().__init__(name, vec, is_atomic=True, struc=struc)
        self.is_concept = is_concept
        self.is_existential = is_existential
        
    @property
    def vec(self):
        if self.A is None: return None
        return self.A[:,self.idx]
    
    def set_vec(self, A, i):
        self.idx = i
        self.A = A
        
    def clear_memory(self):
        self.A = None
        self.idx = None
        
    def __merge_name_or(self, other):
        if isinstance(other, AtomicAxiom) and other.rel == self.rel:
            return f"{Sym.EXISTS.value}{self.rel}.({self.obj}{Sym.CUP.value}{other.obj})"
        return super().__merge_name_or(other)
    
    def __merge_name_and(self, other):
        if isinstance(other, AtomicAxiom) and other.rel == self.rel:
            return f"{Sym.EXISTS.value}{self.rel}.({self.obj}{Sym.CAP.value}{other.obj})"
        return super().__merge_name_and(other)
        
class EmptyAxiom(Axiom):
    def __init__(self):
        name = "(void)"
        struc = {"arity": -1, "op": None, "content": []}
        #vec = np.ones(dim, dtype=bool)
        super().__init__(name, vec=None, is_atomic=True, struc=struc)
        
    def __and__(self, other): return other    
    def __or__(self, other): return other        
    def __rand__(self, other): return other    
    def __ror__(self, other): return other    
    def __invert__(self): return self 
    
    def coverage(self, mask=None): return 0.0    
    def specificity(self, mask=None): return 0.0    
    def xor_score(self, mask): return 0.0        

def extract_from_entity(entity_id, graph, as_string=True, rel_only=False, top_only=False):
    """
    Extract candidate axioms from an entity
    
    Given an entity id from a knowledge graph, retrieve a list of axiom patterns
    that the entity matches. For now, supported patterns are:
    C (raw concept, 'h is a C')
    ∃R.C (it exists x of type C such that (h, r, x) is valid
    
    Make sure you provide the entity identifier, and not the data-specific id of the entity
    (e.g :
    ```
    ent = next(clu[some_cluster].items())
    ent_id = data.indices[ent]
    axioms = extract_from_entity(ent_id, ...)
    ```
    rel_only: discard 'h is a C' relations
    top_only: restrict existential restrictions to ∃R.T (instead of ∃R.C for C =/= top)
    """
    isa = graph.rel.to_id("rdf:type")
    patterns = set()
    if "owl:Thing" not in graph.ent:
        root_object = None
        top_only = False
    else:
        root_object = graph.ent.to_id("owl:Thing")
    for r, ts in graph._h[entity_id].items():
        if r == isa:
            if rel_only: continue
            # Extracting (h, is a, t) triples
            for t in ts:
                name = graph.ent.to_name(t)
                if "dbo:" in name:
                    patterns.add((r, t))
        else:
            if top_only:
                patterns.add((r, root_object))
                continue
            for t in ts:
            # For all x s.t. (h, r, x), find all (x, is a, C) triples
            # and create axiom ∃r.C
                for _, _, c in graph.find_triples(h=t, r=isa, as_string=True):
                    if "dbo:" in c or "owl:" in c:
                        patterns.add((r, graph.ent.to_id(c)))
    if as_string:
        # Convert to strings
        return {(graph.rel.to_name(r), graph.ent.to_name(c)) for r, c in patterns}
    return patterns


def extract_axioms(*args, **kwargs):
    return {AtomicAxiom(a, b) for a,b in extract_from_entity(*args, **kwargs)}


def extract_types_from_entity(entity_id, graph, as_string=True):
    isa = graph.rel.to_id("rdf:type")
    types = set()
    if as_string: 
        def is_valid_type(t): return "dbo:" in t
    else: 
        def is_valid_type(t): return "dbo:" not in graph.ent.to_name(t)
    for _, _, t in graph.find_triples(h=entity_id, r=isa, as_string=as_string):
        if is_valid_type(t):
            types.add(t)
    return types

def extract_types_from_cluster(cluster, graph, data, threshold=0.1, as_freq=True, as_string=True, max_types=None):
    types = Counter()
    for entity in cluster.items():
        entity = data.indices[entity]
        new_types = extract_types_from_entity(entity, graph, as_string=as_string)
        types.update(new_types)
    if threshold is not None:
        threshold = cluster.size * threshold
        types = Counter({t:c for t, c in types.items() if c > threshold})
    if as_freq:
        n = cluster.size
        for t in types: types[t] /= n
    if max_types is not None:
        return Counter({t:c for t,c in types.most_common(max_types)})
    return types

def extract_from_cluster(cluster, graph, data, threshold=0.1, as_freq=True, **kwargs):
    axioms = Counter()
    
    for entity in cluster.items():
        entity = data.indices[entity]
        new_patterns = extract_from_entity(entity, graph, **kwargs)
        axioms.update(new_patterns)        
    if threshold is not None:
        threshold = cluster.size * threshold
        axioms = Counter({ax:ax_count for ax, ax_count in axioms.items() if ax_count>threshold})
    if as_freq:
        n = cluster.size
        for ax in axioms:
            axioms[ax] /= n
    return axioms

def to_string(ax):
    if isinstance(ax, str):
        # Axiom is already a string: we do nothing
        return ax
    pref = ""
    if len(ax) == 3:
        a, b, c = ax
        if c: pref = "¬"
    else:
        a, b = ax
    
    b = b.split(":")[-1]
    if a == "rdf:type":
        return pref+b
    if b == "Thing":
        b = "⊤"
    a = a.split(":")[-1]
    return f"{pref}∃{a}.{b}"