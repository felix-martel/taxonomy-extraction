"""
Contains 'Inducer' the main class for axiom induction


"""

from collections import defaultdict, Counter
from importlib import reload
from itertools import zip_longest, chain
from libs.axiom_induction.patterns import Axiom, AtomicAxiom, EmptyAxiom, Sym
from libs.dataset import Dataset, DEFAULT_EMBED_FILE
from libs.graph import KnowledgeGraph
from libs.table import display_table
from libs.timer import Timer
from libs.utils import Mapper, display_table
from scipy.spatial.distance import cosine as cosine_distance
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm
import itertools as it
import libs.axiom_induction.patterns as patterns
import libs.clustering as lclu
import libs.table as libtab
import libs.taxonomy as tx
import numpy as np
import operator
import os
import pandas as pd
import random
import re

UNARY_OPS = {
    "neg": lambda a:~a,
    "identity": lambda a:a
}

BINARY_OPS = {
    "or": lambda a,b:a|b,
    "and": lambda a,b:a&b
}

class ResultDict(defaultdict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_factory = list
        
    def _ipython_display_(self):
        cols = ["axiom cov spe sco".split()]
        for k, v in self.items():
            cols.append([f"<b>{k}</b>"])
            for i in v:
                cols.append(i)
        display_table(cols)
        
    def flatten(self):
        for step, records in self.items():
            for record in records:
                yield record
        
    def iter_records(self):
        for step, records in self.items():
            for (axiom, cov, spe, sco) in records:
                yield dict(axiom=axiom, cov=cov, spe=spe, sco=sco, step=step)
                
    def iter_axioms(self):
        for axiom, *args in self.flatten():
            yield axiom
            
    def best(self, exclude_concepts=False):
        def key(record):
            ax = record["axiom"]
            if exclude_concepts and ax.is_atomic and ax.is_concept:
                return -float("inf")
            return record["sco"]
        if self:
            return max(self.iter_records(), key=key)
        return None
    
    def pos_only(self, exclude_concepts=False):
        results = ResultDict()
        for step, records in self.items():
            for record in records:
                ax = record[0]
                if not ax.name.startswith(Sym.NEG.value) and (not exclude_concepts or not (ax.is_atomic and ax.is_concept)):
                    results[step].append(record)
        return results
    
    def neg_only(self, exclude_concepts=False):
        results = ResultDict()
        for step, records in self.items():
            for record in records:
                ax = record[0]
                if ax.name.startswith(Sym.NEG.value) and (not exclude_concepts or not (ax.is_atomic and ax.is_concept)):
                    ax.name = ax.name[1:]
                    ax._vec = ax.vec
                    ax.structure = ax.structure["content"][0]
                    results[step].append(record)
        return results
    
    def n_best(self, n, exclude_concepts=False):
        def discard_record(record):
            ax = record["axiom"]
            if exclude_concepts and (
                (ax.is_atomic and ax.is_concept) 
                or (ax.structure["arity"] == 0 and ax.structure["content"][0][0] == "rdf:type")
            ):
                return True
            return False
        records = [rec for rec in sorted(self.iter_records(), key=lambda rec:-rec["sco"])
                   if not discard_record(rec)]
        if n is None:
            return records
        return records[:n]
    
    def ranked(self, limit=None):
        records = sorted(self.flatten(), key=lambda rec:-rec[-1])
        if limit is None:
            return records
        return records[:limit]

class Inducer:
    def __init__(self, clustering, graph, data, rel_only=False, top_only=False, start=None, reverse=False, verbose=True, 
                 extraction_params={}, prefilter_threshold=0.1):
        self.clu = clustering
        self.graph = graph
        self.data = data
        self.verbose = verbose
        self.extraction_params = dict(rel_only=rel_only, top_only=top_only)
        #
        self.c_root = self.clu.root if start is None else self.clu[start]
        ca, cb = self.c_root.children
        if reverse:
            ca, cb = cb, ca
        na, nb = ca.size, cb.size
        self.na, self.nb = na, nb
        entities = [*ca.items(), *cb.items()]
        
        self.A, self.axioms = self.build_axiom_matrix(entities, threshold=prefilter_threshold, **extraction_params)
        self.maska = np.concatenate([np.ones(na, dtype=bool), np.zeros(nb, dtype=bool)])
        self.n_entities, self.n_axioms = self.A.shape
        
    def __repr__(self):
        return f"Inducer(start={self.c_root.name})"
        
    @property
    def maskb(self):
        return ~self.maska
    
    @property
    def n(self):
        return self.na + self.nb
    
    def vprint(self, *args, **kwargs):
        if not self.verbose: return
        print(*args, **kwargs)
        
    def vec(self, axiom):
        return self.A[:, self.axioms.to_id(axiom)]
        
    def as_axiom(self, a, b, is_neg=False):
        """From an unpacked axiom, return the boolean vector associated with it DEPREC"""
        vec = self.vec((a,b))
        if is_neg:
            return ~vec
        return vec
    
    def score(self, axiom, metric="xor"):
        """Return coverage, separation and score of a boolean axiom"""
        return axiom.score(self.maska, metric)
        
    def extract_patterns(self, entity):
        index = self.data.indices[entity]
        return patterns.extract_axioms(index, self.graph, **self.extraction_params)
    
    def old_build_axiom_matrix(self, entities):
        """
        Build an entity-axiom matrix A
        one column per axiom, one line per entity
        A[i, j] = 1 iff axiom a_j holds for entity e_i
        Also return an axiom list (mapping axioms to their ids)
        """
        extracted = [self.extract_patterns(ent) for ent in entities]
        axioms = Mapper({a: i for i, a in enumerate(set(it.chain(*extracted)))}, "axiom", "id")

        A = np.zeros((len(entities), len(axioms)), dtype=bool)
        for i, (ent, axs) in enumerate(zip(entities, extracted)):
            for ax in axs:
                j = axioms.to_id(ax)
                A[i, j] = 1
        for axiom, i in axioms:
            axiom.set_vec(A, i)
        return A, axioms
    
    def ent_extraction(self, ent, individuals=True, existential=True, concepts=True):
        entid = self.data.indices[ent]
        isaid = self.graph.rel.to_id("rdf:type")
        extracted = set()
        excluded_rels = {"rdfs:label", "foaf:name", "dcterms:description"}
        for h, r, t in self.graph.find_triples(h=entid, as_string=True):
            if r in excluded_rels:
                continue
            if r == "rdf:type":
                if not concepts or "dbo:" not in t or "Wikidata:" in t: continue
                extracted.add(AtomicAxiom(r, t))
            elif existential:
                if individuals:
                    extracted.add(AtomicAxiom(r, "{" + t + "}"))
                for _, _, t in self.graph.find_triples(h=self.graph.ent.to_id(t), r=isaid, as_string=True):
                    if "dbo:" not in t and t != "owl:Thing": continue
                    extracted.add(AtomicAxiom(r, t))
        return extracted
    
    def build_axiom_matrix(self, entities, threshold=0.1, **kwargs):
        """
        Build an entity-axiom matrix A
        one column per axiom, one line per entity
        A[i, j] = 1 iff axiom a_j holds for entity e_i
        Also return an axiom list (mapping axioms to their ids)
        """
        min_count = np.floor(threshold * self.n)
        extracted = []
        counts = Counter()
        for ent in entities:
            axioms = self.ent_extraction(ent, **kwargs)
            extracted.append(axioms)
            counts.update(axioms)
        valid_axioms = {ax for ax, count in counts.items() if count > min_count}
        a2i = {a: i for i, a in enumerate(valid_axioms)}
        axioms = Mapper(a2i, "axiom", "id")
        
        A = np.zeros((len(entities), len(axioms)), dtype=bool)
        for i, axs in enumerate(extracted):
            for ax in axs & valid_axioms:
                j = axioms.to_id(ax)
                A[i, j] = 1
        for axiom, i in axioms:
            axiom.set_vec(A, i)
        
        return A, axioms
    
    def generate_candidates(self, start, operators, allow_neg=True, excluded_axioms=set()):
        funcs = {"or": operator.or_, "and": operator.and_}
        for op in operators:
            for axiom in self.axioms.axioms:
                if axiom in excluded_axioms or str(axiom) in excluded_axioms: continue
                yield funcs[op](start, axiom), axiom
                if allow_neg:
                    yield funcs[op](start, ~axiom), axiom
                    
    def filter_similar_axioms(self, axioms):
        # Input: list of {"axiom": ..., "atom": ..., "cov"/"spe"/"sco"/"gain": ...}
        best = defaultdict(int)
        for res in axioms:
            ax = res["atom"]
            if isinstance(ax, AtomicAxiom) and ax.rel != "rdf:type":
                best[ax.rel] = max(best[ax.rel], res["sco"])
        filtered = []
        for res in axioms:
            ax = res["atom"]
            if not isinstance(ax, AtomicAxiom) or ax.rel == "rdf:type" or best[ax.rel] == res["sco"]:
                filtered.append(res)
        return sorted(filtered, key=lambda x:-x["sco"])                
                    
    def improve(self, axiom, keep_n=5, metric="xor", 
                threshold=0.85, cov_threshold=None, spe_threshold=None, 
                allow_neg=True, excluded_axioms=set()):
        icov, ispe, isco = self.score(axiom, metric)
        if isinstance(axiom, EmptyAxiom) or axiom.structure["arity"] < 0:
            icov, ispe, isco = -1, -1, -1
        if cov_threshold is None: cov_threshold = threshold
        if spe_threshold is None: spe_threshold = threshold
        ops = []
        if icov < cov_threshold: ops.append("or")
        if ispe < spe_threshold: ops.append("and")
        def to_results(ax, at):
            cov, spe, sco = self.score(ax, metric)
            gain = sco - isco
            return dict(axiom=ax, atom=at, cov=cov, spe=spe, sco=sco, gain=gain)
        
        results = [to_results(ax, at) 
                   for ax, at in self.generate_candidates(axiom, ops, allow_neg, excluded_axioms)]
        results = self.filter_similar_axioms(results)
        return results
    
    def find(self, max_axioms=3, min_gain=0.05, keep_n=5, forbidden_axioms=set(), **kwargs):
        results = ResultDict()
        step = 0
        axioms = set()
        to_improve = [(EmptyAxiom(), set())]
        while step < max_axioms:
            new_axioms = []
            for axiom, excluded_axioms in to_improve:
                res = self.improve(axiom, excluded_axioms=excluded_axioms|forbidden_axioms, **kwargs)[:keep_n]
                for rec in res:
                    if rec["gain"] < min_gain or rec["axiom"] in axioms:
                        continue
                    atom_used = rec["atom"]
                    rec["atom"] = excluded_axioms | {atom_used}
                    axioms.add(rec["axiom"])
                    new_axioms.append(rec)
            if not new_axioms:
                return results
            new_axioms = sorted(new_axioms, key=lambda x:-x["sco"])[:keep_n]
            results[step] = [(rec["axiom"], rec["cov"], rec["spe"], rec["sco"]) for rec in new_axioms]
            to_improve = [(rec["axiom"], rec["atom"]) for rec in new_axioms]
            step += 1
        return results