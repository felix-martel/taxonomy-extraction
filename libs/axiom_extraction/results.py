from collections import defaultdict, Counter
from functools import total_ordering

from ..utils.display import display_table
import libs.axiom as ax

@total_ordering
class AxiomRecord:
    FIELDS = ["axiom", "cov", "spe", "sco", "step"]
    def __init__(self, axiom, cov, spe, sco, step=-1):
        self.axiom = axiom
        self.cov = cov
        self.spe = spe
        self.sco = sco
        self.step = step

    def as_list(self):
        return [self.axiom, self.cov, self.spe, self.sco, self.step]

    def as_dict(self):
        return {k: getattr(self, k) for k in self.FIELDS}

    def __getitem__(self, item):
        return self.as_list[item]

    def __lt__(self, other):
        if isinstance(other, AxiomRecord):
            return self.sco > other.sco
        return NotImplemented

    def __eq__(self, other):
        return self.sco == other.sco


class ResultDict(defaultdict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_factory = list

    def _ipython_display_(self):
        cols = ["axiom cov spe sco".split()]
        for k, v in self.items():
            cols.append([f"<b>{k}</b>"])
            for i in v:
                cols.append(i.as_list())
        display_table(cols)

    def flatten(self, exclude_concepts=True, exclude_neg=False, exclude_pos=False, exclude_composed=False, exclude_custom=None):
        def is_valid(rec):
            axiom = rec.axiom
            return not (
                (exclude_concepts and ax.is_concept(axiom))
                or (exclude_neg and ax.is_neg(axiom))
                or (exclude_pos and ax.is_pos(axiom))
                or (exclude_composed and ax.is_composed(axiom))
                or (exclude_custom is not None and exclude_custom(axiom))
            )

        for step, records in self.items():
            for record in records:
                if is_valid(record):
                    yield record

    def iter_records(self):
        for rec in self.flatten():
            yield rec.as_dict()

    def iter_axioms(self):
        for rec in self.flatten():
            yield rec.axiom

    def ranked(self, **filters):
        """
        Return a list of records, ranked (highest scores first)
        :param filters: filters for excluding specific types of axioms. The list of valid keyword arguments can be
        found in the documentation of method `ResultDict.flatten()`
        """
        return list(reversed(sorted(self.flatten(**filters))))

    def best(self, **filters):
        """
        Return record with the highest score (among those verifying the provided `filters`)
        :param filters: filters for excluding specific types of axioms. The list of valid keyword arguments can be
        found in the documentation of method `ResultDict.flatten()`
        :return: the `AxiomRecord` with the highest score
        """
        records = self.flatten(**filters)
        if records:
            return max(records)
        return None

    def n_best(self, n=None, **filters):
        records = self.ranked(**filters)
        if n is None or n > len(records):
            return records
        return records[:n]

    def pos_only(self, **filters):
        filters["exclude_neg"] = True
        filters["exclude_pos"] = False

        results = ResultDict()
        for rec in self.flatten(**filters):
            results[rec.step].append(rec)

        return results

    def neg_only(self, **filters):
        filters["exclude_neg"] = False
        filters["exclude_pos"] = True

        results = ResultDict()
        for rec in self.flatten(**filters):
            results[rec.step].append(rec)

        return results