import random
from typing import Optional, Union, List

from .dataset import Dataset
from ..graph import KnowledgeGraph


def create_from_classes(graph: KnowledgeGraph, classes: List[str], class_size: Union[int, List[int]] = 100,
                        force_size: bool = True, **kwargs) -> "Dataset":
    """Create dataset from a list of classnames"""
    # TODO: add proper axiom creation
    cls2name = {cls: name for cls, name in enumerate(classes)}
    name2cls = {name: cls for cls, name in enumerate(cls2name)}

    if isinstance(class_size, int):
        class_size = [class_size] * len(cls2name)
    else:
        if len(class_size) != len(cls2name):
            raise ValueError(f"Size mismatch between class list ({len(cls2name)}) and size list ({len(class_size)})")

    indices = []
    labels = []
    for size, (t, label) in zip(class_size, name2cls.items()):
        ids = graph.sample_instances(size, from_type=t, force_size=force_size, **kwargs)
        labs = [label] * len(ids)
        indices.extend(ids)
        labels.extend(labs)

    return Dataset(indices, labels, name2cls, cls2name, axioms=set())


def create_from_instances(graph: KnowledgeGraph, instances: List[int], is_valid: Optional[bool] = None,
                          isa: str = "rdf:type") -> "Dataset":
    if is_valid is None:
        def is_valid(cls):
            return "dbo:" in cls
    elif isinstance(is_valid, (list, set, tuple)):
        _is_valid = set(is_valid)

        def is_valid(cls):
            return cls in _is_valid
    else:
        if not callable(is_valid):
            raise TypeError("'is_valid' must be a list of valid class names, a function str->bool or None")
    indices = []
    labels = []
    name2cls, cls2name = {}, {}
    r = graph.rel.to_id(isa)
    for i in instances:
        types = {t for _, _, t in graph.find_triples(h=i, r=r, as_string=True) if is_valid(t)}
        if not types:
            continue
        t = random.sample(types, 1)[0]
        if t not in name2cls:
            name2cls[t] = len(name2cls)
        t = name2cls[t]
        indices.append(i)
        labels.append(t)
    cls2name = {c: n for n, c in name2cls.items()}
    # print(len(indices), len(name2cls))
    return Dataset(indices, labels, name2cls, cls2name, axioms=set())
