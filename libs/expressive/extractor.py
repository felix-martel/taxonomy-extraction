"""
Usage: something like
```
extractor = ExpressiveExtractor(kg, params)
extractor.init()

while not extractor.done:
    extractor.next()

"""
from collections import Counter

from libs import embeddings
from libs.axiom_induction.inducer import Inducer
from ..axiom import TopAxiom, RemainderAxiom
from ..tree import Node
from ..utils import Timer, Params
from ..dataset import create_from_instances as create_dataset
from ..cluster import clusterize

import os
import logging
import datetime as dt

import numpy as np

logging.basicConfig()
class ExpressiveExtractor:
    def __init__(self, graph, params, logger=None):
        self.kg = graph
        self.params = self.init_params(params)
        self.name = self.params.record.taxname

        root = TopAxiom

        self.unprocessed = [root]
        self.T = Node(root)
        self.used = set()
        # Mapping axiom --> score
        self.scores = {root: 1.0}
        # Mapping axiom --> number of entities verifying the axiom
        self.sizes = {root: len(self.kg.ent)}
        # Mapping axiom --> depth
        self.depths = dict()
        # Short names of axioms
        self.short_names = dict()
        # Number of searches for a given axiom
        self.n_searches = Counter()
        # Classes for which the search is done
        self.done_classes = set()

        self.E = None

        self.timer = None
        self.logger = self.setup_loggers()

        self._done = False
        self.n_clustering_steps = 0
        self.max_depth = 0

    @property
    def done(self):
        """
        True iff the expressive taxonomy extraction is done
        TODO: for now, we only consider the non-adaptative case, ie we're done when the 'unprocessed' queue is empty
        """
        return not self.unprocessed


    def init_params(self, params):
        params.record.taxname = params.record.name_pattern.format(timestamp=dt.datetime.now(), **params)
        params.record.directory = os.path.join(params.record.dirname, params.record.taxname)

        if params.record.save_taxonomy:
            new_path = params.record.directory
            try:
                os.makedirs(new_path)
            except FileExistsError as e:
                print(f"Directory '{new_path}' already exists.")
                if input("Override ? y/[n]") == "y":
                    os.makedirs(new_path, exist_ok=True)
                else:
                    raise (e)
        return params
    def setup_loggers(self):
        main = logging.getLogger(self.name)
        # File logger (level=DEBUG)
        logger = logging.FileHandler(os.path.join(self.params.record.directory, "log.txt"))
        logger.setLevel(logging.DEBUG)
        main.addHandler(logger)

        # Console logger (level=INFO)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        main.addHandler(console)

        return main



    def init(self):
        self.timer = Timer()
        self.E = embeddings.load(self.params.embeddings)
        self.n_clustering_steps = 0
        self.max_depth = self.params.max_depth


    def get_taxonomy(self):
        pass

    def get_start_axiom(self):
        if self.params.sort_axioms:
            pass
        return self.unprocessed.pop(0)

    def sample_from(self, axiom, size):
        self.sizes[axiom] = size
        return [], size

    def end_search_for(self, axiom, motive="UNK"):
        """
        End search for a given axiom (this axiom won't be used again for sampling/clustering)
        """
        self.done_classes.add(axiom)

         # Logging
        detail = {
            "UNK": "",
            "MIN_SIZE": "not enough instances",
            "MAX_DEPTH": "max recursion depth reached",
            "TAX_DEPTH": "max taxonomic depth reached"
        }.get(motive, "")

        record = f"DEFSTOP/{motive} search done for axiom {axiom}"
        if detail:
            record += f" ({detail})"

        self.logger.debug(record)

    def clusterize(self, instances):
        data = create_dataset(self.kg, instances)
        clu = clusterize(data, self.E, **self.params.clustering)

        self.n_clustering_steps += 1
        return clu

    def extract(self, cluster, reverse=False):
        r = Inducer(start=cluster, graph=self.kg)


    def label_tree(self, root, clu):
        found = set()
        rem = RemainderAxiom(root)
        min_sco, max_sco = -float("inf"), float("inf")

        unvisited = [clu.root]
        search_done = True
        while unvisited:
            node = unvisited.pop()
            if node.is_leaf or node.depth >= self.max_depth or node.size < self.params.halting.min_size:
                # TODO: distinguish between min cluster size and min axiom size
                search_done = False
                continue

            for c, reverse in zip(node.children, [False, True]):
                if c.size < self.params.halting.min_size or c.depth >= self.params.max_depth:
                    search_done = False
                    continue
                label = self.extract(node, reverse)

        if not search_done:
            found.add(RemainderAxiom(root))




    def next(self):
        """
        Run one loop of the algorithm, that is one cycle:
        - choose axiom A
        - sample entities from A
        - clusterize these entities
        - label the clustering tree with new axioms B1, B2, ..., Bk
        - add Bis to the clustering tree
        :return:
        """
        # CHOOSE: Here, start is the start axiom A (from which we'll sample entities) and parent is the parent axiom
        # (to which we'll attach newfound axioms)
        start = self.get_start_axiom()
        parent = start.base if isinstance(start, RemainderAxiom) else start
        # TODO: check max. taxonomic depth
        if self.n_searches[parent] > self.params.halting.max_rec_steps:
            self.end_search_for(parent, "MAX_DEPTH")
            return

        # SAMPLE: Sample instances from the chosen axiom
        instances, n = self.sample_from(start, self.params.size.size)
        self.n_searches[parent] += 1
        if n < self.params.halting.min_size:
            self.end_search_for(parent, "MIN_SIZE")
            return

        # CLUSTER: Clusterize instances' embeddings
        clu = self.clusterize(instances)

        # LABEL: Extract axioms from the clustering tree
















