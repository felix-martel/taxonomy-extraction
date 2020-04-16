from libs.tree import Node
from libs.taxonomy import registered
from typing import List


class Taxonomy(Node):
    """
    Represent a taxonomic tree, that is a graph whose vertices are classes, and edges are subsumption axioms between
    these classes. Compared to its base class `libs.tree.Node`, `Taxonomy` adds a few methods and attributes:
    - `hierarchy` represent the full name of the node;
    - `branch` is similar to `hierarchy`, but it returns a list instead of a string;
    - `clip_at()` for a class equivalent of Node at a given depth;
    Taxonomy also have a few aliases, `from_axioms` and `to_axioms`, as well as a loader, which can be used to
    load pre-registered taxonomies

    Example:
    ```
    >>> T = Taxonomy.load() # Default: load the full DBpedia ontology
    >>> print(T.summary())
    589 classes
    Level 0: owl:Thing
    Level 1: dbo:ChemicalSubstance, dbo:UnitOfWork, dbo:List, dbo:Award, dbo:Event and 36 others
    Level 6: dbo:Guitarist, dbo:SpeedwayRider, dbo:FormulaOneRacer, dbo:NascarDriver, dbo:RaceHorse and 7 others
    Level 7: dbo:FormerMunicipality
    >>> T["dbo:Sport"].print()
              ┌dbo:Boxing
     dbo:Sport┤
              └dbo:Athletics
    >>> t = T["dbo:SoccerPlayer"]
    >>> t.hierarchy
    'owl:Thing/dbo:Agent/dbo:Person/dbo:Athlete/dbo:SoccerPlayer'
    >>> t.branch
    [Taxonomy(owl:Thing),
     Taxonomy(dbo:Agent),
     Taxonomy(dbo:Person),
     Taxonomy(dbo:Athlete),
     Taxonomy(dbo:SoccerPlayer)]
    >>> t < T["dbo:Athlete"]
    True
    >>> t.clip_at(-1), t.clip_at(1), t.clip_at(7)
    (Taxonomy(dbo:SoccerPlayer), Taxonomy(dbo:Agent), Taxonomy(dbo:SoccerPlayer))
    >>> T["dbo:Boxer"] in t
    True
    ```
    """
    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.at_depth = None

    def build_at_depth(self):
        self.at_depth = dict()
        for t in self:
            if t.depth not in self.at_depth:
                self.at_depth[t.depth] = list()
            self.at_depth[t.depth].append(t)

    @classmethod
    def load(cls, name: str = "full") -> "Taxonomy":
        """
        Load a taxonomy from file, using a registered taxonomy (in `libs.taxonomy.registered`) or from a filename
        """
        filename = registered.get(name, name)
        return cls.from_file(filename)

    @property
    def hierarchy(self) -> str:
        return "/".join(cls.name for cls in self.branch)

    @property
    def branch(self) -> List["Taxonomy"]:
        """Return the unique branch going from the root to `self`"""
        return list(self.iter_branch())[::-1]

    def is_subclass_of(self, other: Node) -> bool:
        """
        Return True iff self is a subclass of other.

        Essentially a wrapper around method 'self.__lt__'
        """
        return self < other

    def clip_at(self, depth: int) -> "Taxonomy":
        """Return the class-equivalent at depth `depth`

        depth: no clipping if depth < 0

        e.g. we have hierarchy=SoccerPlayer/Athlete/Person/Agent/Thing, then we'll get :
        clip_at(-1): SoccerPlayer
        clip_at(4) : SoccerPlayer
        clip_at(3) : Athlete
        clip_at(2) : Person
        clip_at(1) : Agent
        clip_at(0) : Thing
        """
        branch = self.branch
        if 0 <= depth < len(branch):
            return branch[depth]
        return self

    def summary(self, n_max: int = 5, depth: int = 2) -> str:
        """
        Return a nice summary of the taxonomy, e.g
        """
        if self.at_depth is None:
            self.build_at_depth()
        n_classes = sum(len(classes) for classes in self.at_depth.values())

        def print_level(k):
            if k is None:
                return "..."
            if k in self.at_depth:
                m = len(self.at_depth[k])
                class_list = [cls.name for cls in self.at_depth[k][:min(m, n_max)]]
                prefix = f"Level {k}: "
                class_list = ", ".join(class_list)
                suffix = "" if n_max > m else f" and {m - n_max} others"
                return prefix + class_list + suffix

        if len(self.at_depth) > 2 * depth:
            d_max = 1 + max(self.at_depth.keys())
            to_print = [*range(depth), None, *range(d_max - depth, d_max)]
        else:
            to_print = sorted(self.at_depth.keys())

        return "\n".join([f"{n_classes} classes", *(print_level(k) for k in to_print)])


    from_axioms = Node.from_edges
    to_axioms = Node.to_edges
