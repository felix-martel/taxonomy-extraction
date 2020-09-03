import operator
import numpy as np

from .graph_extraction import extract_axioms
from .results import AxiomRecord, ResultDict
from ..axiom import is_empty, EmptyAxiom


DEFAULT_ALLOW_NEG = False

class Inducer:
    def __init__(self, Epos, Eneg, graph, threshold=0.1, score="arithmetic",
                 individuals=True, existential=True, concepts=True, verbose=False):
        # TODO: reverse
        self.axioms = extract_axioms([*Epos, *Eneg], graph, threshold,
                                     individuals=individuals, existential=existential, concepts=concepts)
        self.mask = np.concatenate([np.ones(len(Epos), dtype=bool), np.zeros(len(Eneg), dtype=bool)])
        self.n_entities = len(Epos) + len(Eneg)
        self.score = score
        self.verbose = verbose

    def __repr__(self):
        return f"Inducer(entities={self.n_entities}, axioms={len(self.axioms)})"

    def log(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def generate_candidates(self, start, operators, allow_neg=DEFAULT_ALLOW_NEG, forbidden=None):
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

    def improve(self, axiom, metric=None, threshold=0.85, cov_threshold=None, spe_threshold=None,
                allow_neg=DEFAULT_ALLOW_NEG, forbidden=None, reverse=False):
        mask = self.mask if not reverse else ~self.mask

        self.log(f"Improving {axiom}...")
        if cov_threshold is None: cov_threshold = threshold
        if spe_threshold is None: spe_threshold = threshold
        if forbidden is None: forbidden = set()
        if metric is None: metric = self.score

        if is_empty(axiom):
            icov, ispe, isco = 0, 0, 0
        else:
            icov, ispe, isco = axiom.evaluate(mask, how=metric)

        ops = []
        if icov < cov_threshold:
            self.log(f"Coverage too low ({icov:.2f}<{cov_threshold:.2f}). Adding OR clauses...")
            ops.append("or")
        if ispe < spe_threshold:
            self.log(f"Specificity too low ({ispe:.2f}<{spe_threshold:.2f}). Adding AND clauses...")
            ops.append("and")

        results = []
        for ax, at in self.generate_candidates(axiom, ops, allow_neg=allow_neg, forbidden=forbidden):
            cov, spe, sco = ax.evaluate(mask, how=metric)
            gain = sco - isco
            results.append(dict(axiom=ax, atom=at, cov=cov, spe=spe, sco=sco, gain=gain))

        # TODO : filter
        self.log(f"...{len(results)} results found")
        return results

    def find(self, max_axioms=3, min_gain=0.05, keep_n=5, forbidden=None, **kwargs):
        self.log("Finding axioms")
        if forbidden is None:
            forbidden = set()

        results = ResultDict()
        step = 0
        axioms = set()
        # List of axioms (and the atomic axioms used for them)
        to_improve = [(EmptyAxiom(), set())]

        while step < max_axioms:
            new_axioms = []
            self.log(f"\nStep {step}/{max_axioms}: {len(to_improve)} axioms to improve")
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