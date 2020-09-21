import itertools as it
from .ttl import iter_files, get_identifier
import random
import os 

from libs.const import TOY_GRAPH
from .io import iter_training_files, get_item_number, split_line, FILES
from .id_mapper import IdMapper
from .register import register, deregister, get_registered
from collections import defaultdict
from tqdm import tqdm

from .uri import load
from ..utils import Mapper


DoubleDict = lambda:defaultdict(lambda:defaultdict(set))


class KnowledgeGraph:
    """
    Represent a Knowledge Graph.
    
    Available mappings:
    h -> r -> t
    r -> h -> t
    t -> r -> h

    Warning: this implementation is NOT memory efficient. Large graphs can consume a lot of memory (~20GB for the full
    DBpedia graph). Reading a graph from files can take some time (~7 to 9 minutes for DBpedia)

    # TODO: replace IdMappers by Mapper
    # TODO: rethink autodispatch (decorator? subclass? singledispatch?)
    """
    files = FILES
    isa = "rdf:type"
    registered = {
        "full": "data/dbpedia/filtered_3M",
        "toy": "data/graph/toy",
    }

    def __init__(self, entities=None, relations=None, dirname=None):
        self._h = DoubleDict()
        self._r = DoubleDict()
        self._t = DoubleDict()
        
        self.ent = entities if entities is not None else IdMapper()
        self.rel = relations if relations is not None else IdMapper()
        
        self.n_triples = 0
        self.dirname = dirname
            
    def add(self, h, r, t):
        self._h[h][r].add(t)
        self._r[r][h].add(t)
        self._t[t][r].add(h)
        self.n_triples += 1
        
    def rel_to_str(self, item):
        return self.rel.to_name(item)
    
    def ent_to_str(self, item):
        return self.ent.to_name(item)

    @property
    def isaid(self):
        return self.rel.to_id(self.isa)
    
    def triple_to_str(self, item):
        h, r, t = item
        return " ".join([
            self.ent_to_str(h),
            self.rel_to_str(r),
            self.ent_to_str(t)
        ])
        
    def add_uris(self, h, r, t, shorten=False):
        h = get_identifier(h, shorten=shorten)
        t = get_identifier(t, shorten=shorten)
        r = get_identifier(r, shorten=shorten)
        h = self.ent.to_id(h, insert_if_absent=True)
        t = self.ent.to_id(t, insert_if_absent=True)
        r = self.rel.to_id(r, insert_if_absent=True)
        
        self.add(h, r, t)
        
    def __len__(self):
        return self.n_triples
    
    def __iter__(self):
        for h, rs in self._h.items():
            for r, ts in rs.items():
                for t in ts:
                    yield h, r, t
                    
    def iter_names(self):
        for h, r, t in self:
            h, t = self.ent.to_name(h, t)
            r = self.rel.to_name(r)
            yield h, r, t

    def iter_head_tails(self, rel, as_string=False):
        for h, ts in self._r[rel].items():
            for t in ts:
                if as_string:
                    yield self.ent.to_name(h), self.ent.to_name(t)
                else:
                    yield h, t
            
    def dispatch_triple(self, triple, from_="auto", to_="id"):
        # TODO: return proper exception for invalid input formats
        if isinstance(triple, str):
            h, r, t = triple.split(" ")
            return self.dispatch_triple((h, r, t), from_="name")
        h, r, t = triple
        if from_ == "auto":
            from_ = "name" if isinstance(h, str) else "id"
        if from_ == to_:
            return triple
        if from_ == "name":
            return (self.ent.to_id(h), self.rel.to_id(r), self.ent.to_id(t))
        if from_ == "id":
            return (self.ent.to_name(h), self.rel.to_name(r), self.ent.to_name(t))
            
    def __contains__(self, key):
        h, r, t = self.dispatch_triple(key)
        return (h in self._h) and (r in self._h[h]) and (t in self._h[h][r])

    def get_class_sizes(self):
        """
        Return the number of instances of each class in the graph (as a dict class name --> class size)
        """
        return {self.ent.to_name(cls): len(entities) for cls, entities in self._r[self.isaid]}

        
    @classmethod
    def build_from_ttl(cls, *files, shorten=False):
        if len(files) == 1 and isinstance(files[0], list):
            files = files[0]
        kg = KnowledgeGraph()
        
        for h, r, t, _ in iter_files(files):
            kg.add_uris(h, r, t, shorten=shorten)
        return kg
    
    def to_file(cls, path):
        raise NotImplementedError("")
        
    @classmethod
    def triple_files(cls, d=""):
        return [os.path.join(d, cls.files[t]) for t in ["train", "test", "val"]]
        
    @classmethod
    def from_dir(cls, name, max_triples=float("inf"), verbose=True, exclude_entities=None, exclude_relations=None,
                 remove_invalid_types=False, lightweight=True):
        dirname = get_registered(name)
        if exclude_relations is None:
            exclude_relations = set()
        if exclude_entities is None:
            exclude_entities = set()
        if lightweight:
            exclude_relations |= {"rdfs:label", "foaf:name", "dcterms:description"}
            remove_invalid_types = True
        ent = IdMapper.from_file(os.path.join(dirname, cls.files["ent"]), verbose=False)
        rel = IdMapper.from_file(os.path.join(dirname, cls.files["rel"]), verbose=False)
        kg = cls(ent, rel, dirname=dirname)
        
        def is_valid(t):
            name = ent.to_name(t)
            return t == "owl:Thing" or (name.startswith("dbo:") and ":Wikidata" not in name)
        exclude_relations = {rel.to_id(r) for r in exclude_relations if r in rel}
        exclude_entities = {ent.to_id(e) for e in exclude_entities if e in ent}
        isaid = rel.to_id("rdf:type")
        
        n_triples = 0
        expected_n_triples = min(max_triples, get_item_number(cls.triple_files(dirname)))
        for h, r, t, _ in tqdm(iter_training_files(cls.triple_files(dirname)),
                               total=expected_n_triples,
                               disable=not verbose,
                               desc="Triples"):
            if h in exclude_entities or t in exclude_entities or r in exclude_relations:
                continue
            if r == isaid and remove_invalid_types and not is_valid(t):
                continue
            
            if n_triples >= max_triples:
                print("Max number of triples reached")
                return kg
            kg.add(h, r, t)
            n_triples += 1
        return kg

    @classmethod
    def from_triples(cls, triples):
        kg = cls()
        for triple in triples:
            kg.add_uris(*triple)
        return kg
        
    def to_dir(self, dirname, test_split=0.1, val_split=0.1):
        train, test, val = [], [], []
        train_split = 1 - (test_split + val_split)
        
        if not os.path.exists(dirname):
            os.makedirs(dirname)
            
        for h, r, t in self:
            triple = (h, t, r)
            p = random.random()
            if p > 1 - test_split:
                test.append(triple)
            elif p < train_split:
                train.append(triple)
            else:
                val.append(triple)
        
        for n, im in zip(
            ["ent", "rel", "train", "test", "val"], 
            [self.ent, self.rel, train, test, val]
        ):
            with open(os.path.join(dirname, self.files[n]), "w", encoding="utf8") as f:
                print(len(im), file=f)
                for item in im:
                    print(*item, file=f)
        if self.dirname is None:
            self.dirname = dirname

    @classmethod
    def _find(cls, q, db):
        results = defaultdict(lambda:-1)
        qset = set(q)
        for uri, idx in db:
            if uri in qset:
                results[uri] = idx
        return [results[x] for x in q]

    def find_ents(self, q):
        return self._find(q, self.ent)

    def find_rels(self, q):
        return self._find(q, self.rel)
    
    def contains(self, h, r, t):
        return h in self._h and r in self._h[h] and t in self._h[h][r]

    def register(self, name, dirname=None):
        if self.dirname is None and dirname is None:
            raise ValueError("Can't register a knowledge graph without a dirname. Please provide the directory name "
                             "to register with parameter 'dirname'.")
        dirname = self.dirname if dirname is None else dirname
        register(name, dirname)
    
    def print_relations(self, name=None, idx=None, inverse=True):
        if name is not None:
            idx = self.ent.to_id(name)
        elif idx is not None:
            name = self.ent.to_name(idx)
        else:
            raise ValueError("You must provide a name or an identifier")
        print(name)
        for r, ts in self._h[idx].items():
            r = self.rel.to_name(r)
            if len(ts) > 1:
                print("\t", r)
                for t in ts:
                    t = self.ent.to_name(t)
                    print("\t\t", t)
            else:
                t = next(iter(ts))
                t = self.ent.to_name(t)
                print("\t", r, t)
        if inverse:
            for r, ts in self._t[idx].items():
                r = f"is {self.rel.to_name(r)} of"
                if len(ts) > 1:
                    print("\t", r)
                    for t in ts:
                        t = self.ent.to_name(t)
                        print("\t\t", t)
                else:
                    t = next(iter(ts))
                    t = self.ent.to_name(t)
                    print("\t", r, t)

    def is_uri(self, e):
        return isinstance(e, str)

    def is_id(self, e):
        return isinstance(e, int)
    
    def find_triples(self, h=None, r=None, t=None, as_string=False, max_results=None):
        H = self.ent.to_id(h) if self.is_uri(h) else h
        R = self.rel.to_id(r) if self.is_uri(r) else r
        T = self.ent.to_id(t) if self.is_uri(t) else t
        
        def process(a):
            if as_string:
                a = [(self.ent.to_name(h), self.rel.to_name(r), self.ent.to_name(t)) for h, r, t in a]
            if max_results is not None and max_results < len(a):
                return a[:max_results]
            return a
        
        a = {H, R, T}
        if a == {None}:
            # (*, *, *)
            raise ValueError("You must provide at least one argument to <find_triples>")
        elif None not in a:
            # (h, r, t)
            if self.contains(H, R, T):
                return process([(H, R, T)])
        if H is not None:
            if R is not None:
                # (h, r, *)
                return process([(H, R, t) for t in self._h[H][R]])
            elif T is not None:
                # (h, *, t)
                a = []
                for r, ts in self._h[H].items():
                    for t in ts:
                        if t == T:
                            a.append((H, r, T))
                return process(a)
            else:
                # (h, *, *)
                a = []
                for r, ts in self._h[H].items():
                    for t in ts:
                        a.append((H, r, t))
                return process(a)
        elif R is not None:
            if T is not None:
                # (*, r, t)
                return process([(h, R, T) for h in self._t[T][R]])   
            else:
                # (*, r, *)
                a = []
                for h, ts in self._r[R].items():
                    for t in ts:
                        a.append((h, R, t))
                return process(a)
        else:
            # (*, *, t)
            a = []
            for r, hs in self._t[T].items():
                for h in hs:
                    a.append((h, r, T))
            return process(a)
                
    def sample_instances(self, n, from_type="owl:Thing", except_type=None, type_rel="rdf:type", as_string=False, force_size=True, exclude_ids=None):
        is_a = self.rel.to_id(type_rel)
        t = self.ent.to_id(from_type)
        instances = {h for h, _, _ in self.find_triples(r=is_a, t=t, as_string=as_string)}
        if except_type is not None:
            #t_exc = self.ent.to_id(except_type)
            r, t_excluded = (type_rel, except_type) if as_string else (is_a, self.ent.to_id(except_type))
            instances = {h for h in instances if (h, r, t_excluded) not in self}
        if exclude_ids is not None:
            instances -= exclude_ids
        if len(instances) < n:
            if force_size: 
                raise ValueError(f"Can't sample {n} items from a set of size {len(instances)} for type '{from_type}'.")
            return instances
        return random.sample(instances, n)

    def get_heads(self, t):
        """t --> h"""
        return set(it.chain(*self._t[t].values()))

    def get_tails(self, h):
        """h --> t"""
        return set(it.chain(*self._h[h].values()))

    def get_subjects(self, r):
        """r --> h"""
        return self._r[r].keys()

    def get_rels(self, h):
        """h --> r"""
        return self._h[h].keys()

    get_out_rels = get_rels

    def get_in_rels(self, t):
        """t --> r"""
        return self._t[t].keys()

def load_toyset():
    kg = KnowledgeGraph()
    with open(TOY_GRAPH, "r") as f:
        for line in f:
            h, r, t = line.split()
            kg.add_uris(h, r, t)
    return kg