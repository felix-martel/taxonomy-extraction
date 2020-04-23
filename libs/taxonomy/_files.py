from typing import Dict

# TODO: use config
DBPEDIA_FULL = "data/taxonomy/full.txt"
DBPEDIA_SMALL = "data/taxonomy/small.txt"
DBPEDIA_TOY = "data/taxonomy/toy.txt"

registered: Dict[str, str] = {
    "full": DBPEDIA_FULL,
    "small": DBPEDIA_SMALL,
    "toy": DBPEDIA_TOY
}