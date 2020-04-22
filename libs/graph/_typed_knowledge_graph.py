from .knowledge_graph import KnowledgeGraph


class TypedKnowledgeGraph(KnowledgeGraph):
    def __init__(self, type_rel: str, entities=None, relations=None):
        super().__init__(entities, relations)
        self.isa = type_rel

    @property
    def isaid(self):
        return self.rel.to_id(self.isa)


    def is_a(self, e, t):
        e = self.ent.to_id(e)
        t = self.ent.to_id(t)
        return (e, self.isaid, t) in self

    @property
    def types(self):
        return self._r[self.isaid]

    def get_types(self, entity, as_string=False):
        e = self.ent.to_id(entity)
        ts = self.types[e]
        if as_string:
            return {self.ent.to_name(x) for x in ts}
        return ts

