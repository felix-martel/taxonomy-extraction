import operator
import numpy as np

from .graph_extraction import extract_axioms
from .results import AxiomRecord, ResultDict
from ..axiom import is_empty, EmptyAxiom


class Inducer:
    def __init__(self, A, B, graph, threshold=0.1, score="arithmetic", individuals=True, existential=True, concepts=True):
        # TODO: reverse
        self.axioms = extract_axioms([*A, *B], graph, threshold,
                                     individuals=individuals, existential=existential, concepts=concepts)
        self.mask = np.concatenate([np.ones(len(A), dtype=bool), np.zeros(len(B), dtype=bool)])
        self.n_entities = len(A) + len(B)
        self.score = score

    def __repr__(self):
        return f"Inducer(entities={self.n_entities}, axioms={len(self.axioms)})"

    def generate_candidates(self, start, operators, allow_neg=True, forbidden=None):
        if forbidden is None:
            forbidden = set()

        funcs = {"or": operator.or_, "and": operator.and_}
        for op in operators:
            for axiom, j in self.axioms:
                if axiom in forbidden:
                    continue
                yield funcs[op](start, axiom), axiom
                if allow_neg:
                    yield funcs[op](start, ~axiom), axiom

    def improve(self, axiom, metric=None, threshold=0.85, cov_threshold=None, spe_threshold=None, allow_neg=True, forbidden=None):
        print(f"Improving {axiom}...")
        if cov_threshold is None: cov_threshold = threshold
        if spe_threshold is None: spe_threshold = threshold
        if forbidden is None: forbidden = set()
        if metric is None: metric = self.score

        if is_empty(axiom):
            icov, ispe, isco = 0, 0, 0
        else:
            icov, ispe, isco = axiom.evaluate(self.mask, how=metric)

        ops = []
        if icov < cov_threshold:
            print(f"Coverage too low ({icov:.2f}<{cov_threshold:.2f}). Adding OR clauses...")
            ops.append("or")
        if ispe < spe_threshold:
            print(f"Specificity too low ({ispe:.2f}<{spe_threshold:.2f}). Adding AND clauses...")
            ops.append("and")

        results = []
        for ax, at in self.generate_candidates(axiom, ops, allow_neg=allow_neg, forbidden=forbidden):
            cov, spe, sco = ax.evaluate(self.mask, how=metric)
            gain = sco - isco
            results.append(dict(axiom=ax, atom=at, cov=cov, spe=spe, sco=sco, gain=gain))

        # TODO : filter
        print(f"...{len(results)} results found")
        return results

    def find(self, max_axioms=3, min_gain=0.05, keep_n=5, forbidden=None, **kwargs):
        print("Finding axioms")
        if forbidden is None:
            forbidden = set()

        results = ResultDict()
        step = 0
        axioms = set()
        # List of axioms (and the atomic axioms used for them)
        to_improve = [(EmptyAxiom(), set())]

        while step < max_axioms:
            new_axioms = []
            print(f"\nStep {step}/{max_axioms}: {len(to_improve)} axioms to improve")
            for axiom, excluded_axioms in to_improve:
                res = self.improve(axiom, forbidden=excluded_axioms | forbidden, **kwargs)[:keep_n]
                for rec in res:
                    if rec["gain"] < min_gain or rec["axiom"] in axioms:
                        continue
                    atom_used = rec["atom"]
                    rec["atom"] = excluded_axioms | {atom_used}
                    axioms.add(rec["axiom"])
                    new_axioms.append(rec)
            if not new_axioms:
                return results
            new_axioms = sorted(new_axioms, key=lambda x: -x["sco"])[:keep_n]
            results[step] = [AxiomRecord(rec["axiom"], rec["cov"], rec["spe"], rec["sco"], step) for rec in new_axioms]
            to_improve = [(rec["axiom"], rec["atom"]) for rec in new_axioms]
            step += 1
        return results