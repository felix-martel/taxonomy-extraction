import os
from collections import defaultdict
from tqdm import tqdm

def split_couple(line, sep=" "):
    a = line.rstrip().split(sep)
    idx = a[-1]
    name = sep.join(a[:-1])
    return name, int(idx)

class IdMapper:
    """
    A two-way mapping between URIs and IDs in a knowledge graph.

    Class `utils.Mapper` is based on it
    # TODO: replace it by the generic `utils.Mapper`
    """
    prefixes = {
       "http://dbpedia.org/ontology/": "dbo",
        "http://dbpedia.org/resource/": "dbr",
        "http://purl.org/dc/terms/": "dcterms",
        "http://www.w3.org/2002/07/owl#": "owl",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#": "rdf",
        "http://www.w3.org/2000/01/rdf-schema#": "rdfs",
        "http://xmlns.com/foaf/0.1/": "foaf",
        "http://purl.org/vocab/vann/": "vann",
        "http://www.w3.org/2004/02/skos/core#": "skos",
        "http://purl.org/dc/elements/1.1/": "dce",
        "http://schema.org/": "schema",
        "http://purl.org/vocommons/voaf#": "voaf",
        "http://www.w3.org/2001/XMLSchema#": "xsd",
        "http://www.wikidata.org/entity/": "wd"
    }
    
    def __init__(self, n2i=None, i2n=None, embeddings=None):
        if n2i is None and i2n is None:
            self.uri = {}
            self.idx = {}
        elif i2n is None:
            self.uri = n2i
            self.idx = {i: n for n, i in n2i.items()}
        elif n2i is None:
            self.idx = i2n
            self.uri = {n: i for i, n in i2n.items()}
        else:
            self.uri = n2i
            self.idx = i2n
        self.embeddings = embeddings
            
    def add(self, n, i=None):
        if i is None:
            i = len(self)
        self.uri[n] = i
        self.idx[i] = n

    @classmethod
    def shorten(self, name):
        for url, pref in self.prefixes.items():
            if url in name:
                name = name.replace(url, pref+":").replace("<", "").replace(">", "")
                return name
        return name
    
    def to_name(self, *ids):
        def name(idx): return self.shorten(self.to_uri(idx))
        
        res = [name(idx) for idx in ids]
        if len(res) == 1: return res[0]
        return res
    
    def to_uri(self, idx):
        return self.idx[idx]

    def to_uris(self, *ids):
        return [self.to_uri(idx) for idx in ids]

    def to_ids(self, *uris, insert_if_absent=False):
        return [self.to_id(uri, insert_if_absent) for uri in uris]
    
    def to_id(self, uri, insert_if_absent=False):
        if insert_if_absent and uri not in self.uri:
            new_idx = len(self.uri)
            self.add(uri, new_idx)
        return self.uri[uri]
    
    def idx_to_vec(self, idx):
        return self.embeddings[idx]
    
    def uri_to_vec(self, uri):
        return self.idx_to_vec(self.to_id(uri))
    
    def vec(self, uri):
        return self.uri_to_vec(uri)
        
    def __len__(self):
        return len(self.uri)
    
    def __iter__(self):
        for n, i in self.uri.items():
            yield n, i
            
    def __contains__(self, key):
        return key in self.uri or key in self.idx
    
    def search(self, q, max_results=100):
        results = []
        for uri, idx in self:
            if q in uri:
                results.append((uri, idx))
                if len(results) == max_results:
                    return results
        return results
    
    def search_many(self, q):
        results = defaultdict(lambda:-1)
        qset = set(q)
        for uri, idx in self:
            if uri in qset:
                results[uri] = idx
        return [results[x] for x in q]
    
    def set_embeddings(self, embeddings):
        self.embeddings = embeddings
    
    @classmethod
    def from_file(cls, filename, verbose=True):
        rel = cls()
        with open(filename, "r", encoding="utf8") as f:
            n_items = int(next(f))
            for i, line in tqdm(enumerate(f), total=n_items, disable=not verbose):
                try:
                    uri, idx = split_couple(line)
                except ValueError as e:
                    print(i, line)
                    raise e
                rel.add(uri, idx)
            return rel
