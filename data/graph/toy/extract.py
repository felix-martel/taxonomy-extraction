import os
from libs.graph import KnowledgeGraph

# Classes to sample entities from
CLASSES = ["dbo:Person", "dbo:Organisation", "dbo:Event", "dbo:Place"]
N_ENTITIES = 5000

# Output dir
DIRNAME = "data/dbpedia/toy"

# 1/ Load full graph
kg = KnowledgeGraph.from_dir("data/dbpedia/filtered_3M",
							exclude_relations={"rdfs:label", "foaf:name", "dcterms:description"},
							remove_invalid_types=True,
							max_triples=float("inf")) 
							
isaid = kg.rel.to_id("rdf:type")

# 2/ Sample entities and their neighbors
triples = set()
for cls in CLASSES:
    # Sample 5000 instances from each class, then add to the toy graph all neighboring entities, plus all ISA triples of these neighbors
    instances = kg.sample_instances(N_ENTITIES, cls)
    for h in instances:
        for h, r, t in kg.find_triples(h=h, as_string=True):
            triples.add((h, r, t))
            for a, b, c in kg.find_triples(h=kg.ent.to_id(t), r=isaid, as_string=True):
                triples.add((a, b, c))
            
print(len(triples), "triples sampled")

# 3/ Convert triples to OpenKE format (see https://github.com/thunlp/OpenKE)
ents = dict()
rels = dict()
data = set()

for e1, r, e2 in triples:
    e1 = ents.setdefault(e1, len(ents))
    e2 = ents.setdefault(e2, len(ents))
    r = rels.setdefault(r, len(rels))
    data.add((e1, e2, r))    
	
# 4/ Save dummy graph 
with open(os.path.join(DIRNAME, "entity2id.txt"), "w") as f:
    print(len(ents), file=f)
    for e, i in ents.items():
        print(e, i, file=f)
        
with open(os.path.join(DIRNAME, "relation2id.txt"), "w") as f:
    print(len(rels), file=f)
    for e, i in rels.items():
        print(e, i, file=f)
        
with open(os.path.join(DIRNAME, "train2id.txt"), "w") as f:
    print(len(data), file=f)
    for e1, e2, r in data:
        print(e1, e2, r, file=f)        

