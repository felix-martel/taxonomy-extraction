"""
Usage: something like
```
extractor = ExpressiveExtractor(kg, params)
extractor.init()

while not extractor.done:
    extractor.next()

"""
import logging
import sys
import os
import datetime as dt
from collections import Counter

from libs import embeddings
#from libs.axiom_induction.inducer import Inducer
from libs.axiom_extraction import Inducer as AxiomInducer
from ..axiom import TopAxiom, RemainderAxiom
from ..tree import Node
from ..utils import Timer, Params
from ..dataset import create_from_instances as create_dataset
from ..cluster import clusterize
from ..sampling import GraphSampler


import numpy as np

#logging.basicConfig()

class ExpressiveExtractor:
    def __init__(self, graph, params, sampler=None, verbose=logging.INFO):
        self.kg = graph
        self.params = self.init_params(params)
        self.name = self.params.record.taxname

        root = TopAxiom

        self.unprocessed = [root]
        self.T = Node(root)
        self.used = {root}
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
        self.sampler = GraphSampler(self.kg) if sampler is None else sampler

        self.timer = None
        self.logger = self.setup_loggers(verbose)

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


    @property
    def status(self):
        return f"STATUS\nDone: {self.done}\n"\
            f"Step: {self.n_clustering_steps}\n"\
            f"Time started: {self.get_time_started()}"

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

    def setup_loggers(self, level=logging.INFO):
        main = logging.getLogger(self.name)
        main.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s : %(message)s", datefmt="%H:%M:%S")

        # File logger (level=DEBUG)
        debug_handler = logging.FileHandler(os.path.join(self.params.record.directory, "log.txt"), encoding="utf8")
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(formatter)

        # Console logger (level=INFO)
        main_handler = logging.StreamHandler(sys.stdout)
        main_handler.setLevel(level)
        main_handler.setFormatter(formatter)

        main.addHandler(debug_handler)
        main.addHandler(main_handler)

        main.propagate = False
        return main

    def get_time_started(self):
        if self.timer is None:
            return "not started"
        start = dt.datetime.fromtimestamp(self.timer._start)
        return f"{start:%H:%M:%S}"


    def init(self):
        self.logger.debug("Initialisation started.")
        self.timer = Timer()
        self.E = embeddings.load(self.params.embeddings)
        self.logger.debug(f"Embeddings loaded, matrix shape is {self.E.shape}")

        self.n_clustering_steps = 0
        self.max_depth = self.params.max_depth
        self.logger.info("Initialisation done.")


    def get_taxonomy(self):
        pass

    def get_start_axiom(self):
        if self.params.sort_axioms:
            raise NotImplemented("'get_start_axiom' is not implemented yet when 'sort_axioms' is set to True")
        return self.unprocessed.pop(0)

    def sample_from(self, axiom, size):
        instances, size = self.sampler.sample(axiom, size)
        self.sizes[axiom] = size

        self.logger.debug(f"Sampled {len(instances)} instances out of {size} from concept {axiom}")
        return instances, size

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

        self.logger.debug(f"Clustering done over {len(instances)} embedding vectors. ")
        self.n_clustering_steps += 1
        return clu

    def label_tree(self, parent, clu):
        found = set()
        root = clu.root
        rem = RemainderAxiom(parent)
        min_sco, max_sco = -float("inf"), float("inf")

        self.logger.debug(f"Starting tree labelling for axiom {parent} over {clu.size} clusters.")

        unvisited = [clu.root]
        search_done = True
        while unvisited:
            node = unvisited.pop()
            self.logger.debug(f"Processing cluster {node}: depth={node.depth}, size={node.size}")
            if node.is_leaf or node.depth >= self.max_depth or node.size < self.params.halting.min_size:
                # TODO: distinguish between min cluster size and min axiom size
                search_done = False
                self.logger.debug(f"Search stop for cluster {node} (is leaf or max depth reached or cluster size too low)")
                continue

            E_pos, E_neg = [list(c.items()) for c in node.children]
            inducer = AxiomInducer(E_pos, E_neg, self.kg)
            for c, reverse in zip(node.children, [False, True]):
                if c.size < self.params.halting.min_size or c.depth >= self.params.max_depth:
                    search_done = False
                    self.logger.debug(f"Search stop for cluster {c} (size too low or max depth reached)")
                    continue
                axioms = inducer.find(reverse=reverse, forbidden=self.used)
                label = axioms.best()
                if label is not None:
                    # 'label' has type 'AxiomRecord'
                    self.logger.debug(f"Found axiom {label.axiom} for subcluster {c}")
                    found.add(label.axiom)
                    continue
                else:
                    self.logger.debug(f"No axiom found for subcluster {c}, search continues")
                    unvisited.append(c)

        if not search_done:
            found.add(rem)

        self.logger.info(f"Subclasses found: " + ", ".join(str(x) for x in found))
        return found

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
        self.logger.info(f"STEP {self.n_clustering_steps}: starting with axiom {start}")

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
        labels = self.label_tree(parent, clu)
        self.T[parent].add_many(labels)
        self.used.update(labels)
        self.unprocessed.extend(labels)

        return start, instances, clu, labels

    def run(self, n_runs=None):
        while not self.done and (n_runs is None or n_runs > 0):
            self.next()
            if n_runs is not None:
                n_runs -= 1
















