import random
import os 
import numpy as np

from .knowledge_graph import KnowledgeGraph, DoubleDict, IdMapper


class MemKnowledgeGraph(KnowledgeGraph):
    """
    A more memory-efficient implementation Knowledge Graph.
    
    Available mappings:
    h -> r, t
    """
    files = {
        "rel": "relation2id.txt",
        "ent": "entity2id.txt",
        "triples": "triples2id.txt",
        "train": "train2id.txt", 
        "test": "test2id.txt",
        "val": "valid2id.txt"
    }
    def __init__(self, entities=IdMapper(), relations=IdMapper()):
        self._h = defaultdict(set)
        
        self.ent = entities
        self.rel = relations
        
        self.n_triples = 0
            
    def add(self, h, r, t):
        self._h[h].add((r, t))
        self.n_triples += 1  
    
    def __iter__(self):
        for h, rs in self._h.items():
            for r, t in rs:
                yield (h, r, t)
            
    def __contains__(self, key):
        h, r, t = self.dispatch_triple(key)
        return h in self._h and (r, t) in self._h[h]
    
    def find_triples(self, h=None, r=None, t=None, as_string=False, max_results=None):
        H, R, T = h, r, t
        
        def process(a):
            if not a:
                return a
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
            if (H, R, T) in self:
                return process([(H, R, T)])
            return process([])
        if H is not None:
            if R is not None:
                # (h, r, *)
                return process([(H, R, t) for r, t in self._h[H] if r == R])
            elif T is not None:
                # (h, *, t)
                return process([(H, r, T) for r, t in self._h[H] if t == T])
            else:
                # (h, *, *)
                return process([(H, r, t) for r,t in self._h[H]])
        elif R is not None:
            if T is not None:
                # (*, r, t)
                return process([(h, R, T) for h, r, t in self if r == R and t == T])  
            else:
                # (*, r, *)
                return process([(h, R, t) for h, r, t in self if r == R])  
                #return process(a)
        else:
            # (*, *, t)
            return process([(h, r, T) for h, r, t in self if t == R]) 
            #raise NotImplementedError("Request (*, *, t) is not supported in low memory knowledge graph")
                
    def sample_instances(self, n, from_type="owl:Thing", except_type=None, type_rel="rdf:type", as_string=False, force_size=True, exclude_ids=None):
        t = self.ent.to_id(from_type)
        is_a = self.rel.to_id(type_rel)
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