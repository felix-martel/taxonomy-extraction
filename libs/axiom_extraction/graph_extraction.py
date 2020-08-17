from collections import Counter

import numpy as np

from ..axiom import Concept, Existential, Axiom
from ..utils import Mapper

def ent_extraction(ent, graph, individuals=True, existential=True, concepts=True):
    isaid = graph.rel.to_id("rdf:type")
    extracted = set()
    excluded_rels = {"rdfs:label", "foaf:name", "dcterms:description"}
    for h, r, t in graph.find_triples(h=ent, as_string=True):
        if r in excluded_rels:
            continue
        if r == "rdf:type":
            if not concepts or "dbo:" not in t or "Wikidata:" in t: continue
            extracted.add(Concept(t))
        elif existential:
            if individuals:
                extracted.add(Existential(r, Concept(singleton=t)))

            for _, _, t in graph.find_triples(h=graph.ent.to_id(t), r=isaid, as_string=True):
                if "dbo:" not in t and t != "owl:Thing": continue
                extracted.add(Existential(r, Concept(t)))
    return extracted


def extract_axioms(ents, graph, threshold=0.1, **params):
    n = len(ents)
    min_count = n * threshold
    counts = Counter()
    extracted = []

    for ent in ents:
        axioms = ent_extraction(ent, graph, **params)
        extracted.append(axioms)
        counts.update(axioms)

    valid_axioms = {axiom for axiom, count in counts.items() if count > min_count}
    axioms = Mapper({a: i for i, a in enumerate(valid_axioms)}, "axiom",  "id", auto_id=False, type_a=Axiom, type_b=int)

    A = np.zeros((len(ents), len(axioms)), dtype=bool)
    for i, axs in enumerate(extracted):
        for ax in axs & valid_axioms:
            j = axioms.to_id(ax)
            A[i, j] = 1

    for axiom, i in axioms:
        axiom.vec = A[:, i]

    return axioms