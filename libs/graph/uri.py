from typing import Tuple
from tqdm import tqdm
from libs.utils import Mapper


SEP = ":"
PREFIXES: Mapper[str, str] = Mapper({
    "http://dbpedia.org/ontology/": "dbo",
    "http://dbpedia.org/resource/": "dbr",
    "http://dbpedia.org/datatype/": "dbt",
    "http://www.ontologydesignpatterns.org/ont/d0.owl#": "d0",
    "http://purl.org/dc/terms/": "dcterms",
    "http://www.w3.org/2002/07/owl#": "owl",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#": "rdf",
    "http://www.w3.org/2000/01/rdf-schema#": "rdfs",
    "http://www.w3.org/ns/prov#": "prov",
    "http://xmlns.com/foaf/0.1/": "foaf",
    "http://purl.org/vocab/vann/": "vann",
    "http://www.w3.org/2004/02/skos/core#": "skos",
    "http://purl.org/dc/elements/1.1/": "dce",
    "http://schema.org/": "schema",
    "http://creativecommons.org/ns#": "cc",
    "http://purl.org/vocommons/voaf#": "voaf",
    "http://www.w3.org/2001/XMLSchema#": "xsd",
    "http://www.wikidata.org/entity/": "wd"
}, "long", "short")


def shorten(uri: str) -> str:
    """
    If possible, replace a long URI by its prefix,
    e.g "http://dbpedia.org/ontology/Agent" becomes "dbo:Agent"
    """
    for long, short in PREFIXES:
        if uri.startswith(long):
            return uri.replace(long, short + SEP)
    return uri


def lengthen(uri: str) -> str:
    """
    If possible, replace the URI of a prefix by its long form,
    e.g "dbo:Agent" becomes "http://dbpedia.org/ontology/Agent"
    """
    if SEP in uri:
        pref, uri = uri.split(SEP, maxsplit=1)
        return PREFIXES.to_long(pref) + uri
    return uri


def __split_couple(line: str, sep: str = " ") -> Tuple[str, int]:
    a = line.rstrip().split(sep)
    idx = a[-1]
    name = sep.join(a[:-1])
    return name, int(idx)


def load(filename: str, verbose: bool = False) -> Mapper[str, int]:
    with open(filename, "r") as f:
        n_items = int(next(f))
        u2i = [__split_couple(line) for line in tqdm(f, total=n_items, disable=not verbose)]
    return Mapper(u2i, "uri", "id", type_a=str, type_b=int)
