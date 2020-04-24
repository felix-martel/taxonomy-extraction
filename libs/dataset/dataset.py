import random
from typing import Optional, Iterable, Tuple, Dict, List, Iterator, Set
from collections import defaultdict, Counter

from .io import load_dataset, save_dataset

AxiomTuple = Tuple[str, str]


class Dataset:
    DBPEDIA_SMALL = "data/taxonomies/full_small/"
    DBPEDIA_FULL = "data/taxonomies/full_large/" #"data/full_flat_clusters/"
    REGISTERED_DATASETS = {
        "small": DBPEDIA_SMALL,
        "full": DBPEDIA_FULL,
        "large": DBPEDIA_FULL # alias for 'full'
    }
    
    def __init__(self, indices: List[int], labels: List[int],
                 name2cls: Dict[str, int], cls2name: Dict[int, str],
                 axioms: Iterable[AxiomTuple], dirname: Optional[str] = None,
                 remove_empty_classes: bool = False) -> None:
        self.ids = list(range(len(indices)))
        self.indices = indices
        self.labels = labels
        self.name2cls = name2cls
        self.cls2name = cls2name
        self.axioms: Set[AxiomTuple] = set(axioms)
        self.class_count = self._get_class_count()
        self.class_instances = self._get_class_instances()
        self.dirname = dirname
        if remove_empty_classes:
            self.remove_empty_classes()
        
    def remove_empty_classes(self, verbose=False):
        empty_classes = {cls_name: cls_id for cls_name, cls_id in self.name2cls.items()
                         if cls_name not in self.class_count
                         or self.class_count[cls_name] == 0
                         }
        if verbose:
            print(f"Remove the following empty classes: {', '.join(empty_classes)}")
        for cls_name, cls_id in empty_classes.items():
            del self.name2cls[cls_name]
            del self.cls2name[cls_id]
        self.axioms = {(a, b) for (a, b) in self.axioms if a not in empty_classes and b not in empty_classes}

    @property
    def n_classes(self) -> int:
        """
        Number of unique classes in the dataset
        """
        return len(self.name2cls)

    @property
    def n_instances(self) -> int:
        """
        Number of entities in the dataset
        """
        return len(self.indices)

    @classmethod
    def load(cls, dirname) -> "Dataset":
        """Load dataset for a directory"""
        if dirname in cls.REGISTERED_DATASETS:
            dirname = cls.REGISTERED_DATASETS[dirname]
        data = load_dataset(dirname)
        return cls(**data)
    
    def save(self, dirname):
        """Save dataset to a directory"""
        save_dataset(self, dirname)

    def __iter__(self) -> Iterator:
        """Iterate over (entity id, entity class) pairs"""
        for nid, index, label in zip(self.ids, self.indices, self.labels):
            yield nid, index, label
            
    def __len__(self) -> int:
        """Return the number of entities in the dataset"""
        return len(self.ids)
    
    def _get_class_count(self) -> Counter:
        """Get the number of entities per class"""
        return Counter(self.cls2name[cls] for cls in self.labels)
    
    def _get_class_instances(self) -> Dict[str, set]:
        """Get the list of entity ids in each class"""
        instances = defaultdict(set)
        for instance, index, label in self:
            class_name = self.cls2name[label]
            instances[class_name].add(instance)
        return instances
    
    def sample_from_cls(self, cls_name: str, k: int = 1):
        """Sample `k` entities from a given class"""
        samples = random.sample(self.class_instances[cls_name], k=k)
        if k == 1:
            return samples[0]
        return samples
    
    def summary(self, n=5) -> str:
        """Return a summary of the dataset's content (number of classes, instances & more)"""
        maxlen = max(map(len, self.name2cls.keys())) + 2
        header = f"Dataset ({len(self.name2cls)} classes, {len(self)} instances):\n---\n"
        freqs = "\n".join(f"{cls:{maxlen}} {count}" for cls, count in self.class_count.most_common(n))
        return header + freqs + "\n..."
    
    def set_root(self, root: str):
        children, parents = zip(*self.axioms)
        subroots = set(parents) - set(children)
        if len(subroots) == 1:
            # There's alreay a single root
            return
        for child in subroots:
            if child != root:
                new_axiom = (child, root)
                self.axioms.add(new_axiom)
    