from enum import Enum

class Rel(Enum):
    IS_A = "rdf:type"
    ROOT = "owl:Thing"

    def __str__(self):
        return str(self.value)


class Sym(Enum):
    NEG = "¬"
    EXISTS = "∃"
    TOP = "⊤"
    OR = "∨"
    AND = "∧"
    CUP = "⊔"
    CAP = "⊓"
    IN = "∈"

    def __str__(self):
        return str(self.value)