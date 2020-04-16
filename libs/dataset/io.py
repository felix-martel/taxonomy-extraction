import os
from enum import Enum

class File(Enum):
    INDICES = "cluster.indices"
    LABELS = "cluster.labels"
    NAMES = "name_to_index"
    AXIOMS = "axioms"
    
    def path(self, dirname):
        return os.path.join(dirname, self.value)

def load_instances(dirname):
    """
    Read files 'cluster.indices' and 'cluster.labels' from the disk.
    Return two arrays, containing respectively a list of entitiy indices and a list of their labels.
    
    @param
    dirname: str, directory containing the files
    """
    i = File.INDICES.path(dirname)
    l = File.LABELS.path(dirname)
    with open(i, "r") as fi:
        indices = [int(line.rstrip()) for line in fi]
    with open(l, "r") as fl:
        labels = [int(line.rstrip()) for line in fl]
    return indices, labels

def save_instances(indices, labels, dirname):
    """
    Write lists of indices and labels to file. Create two files 'cluster.indices' and 'cluster.labels' 
    in 'dirname', each containing one integer per line
    """
    i = File.INDICES.path(dirname)
    l = File.LABELS.path(dirname)
    with open(i, "w") as fi:
        for idx in indices:
            print(idx, file=fi)
    with open(l, "w") as fl:
        for lab in labels:
            print(lab, file=fl)

def load_classes(dirname):
    """
    Load 'name_to_index' file. Return two dicts, one mapping names to integer codes, and the other integer codes to names
    
    @params
    dirname: str, path of the directory containing the code
    """
    name2cls = {}
    cls2name = {}
    with open(File.NAMES.path(dirname), "r") as f:
        for line in f:
            name, cls = line.split() #rstrip().split(" ")
            cls = int(cls)
            name2cls[name] = cls
            cls2name[cls] = name
    return name2cls, cls2name

def save_classes(name2cls, dirname):
    """
    Save a mapping from class names to their integer code, as used in 'data.labels'
    File format is one class per line, class and codes separated by a blank space:
    ```
    ClassA 0
    ClassB 1
    ClassC 2
    ...
    ```
    
    @params
    name2cls: dict str->int
    dirname: str, directory path
    """
    with open(File.NAMES.path(dirname), "w") as f:
        for name, label in name2cls.items():
            print(name, label, file=f)

def load_axioms(dirname):
    """
    Load an "axioms" file and return a list of axioms, ie of pair ("foo", "bar") s.t axiom 'foo⊏bar' holds.
    See `save_axioms` doc for file format
    """
    axioms = []
    with open(File.AXIOMS.path(dirname), "r") as f:
        for line in f:
            child, parent = line.split()
            axioms.append((child, parent))
    return axioms

def save_axioms(axioms, dirname):
    """
    Save a list of axioms. File format: one axiom per line, child and parent class names separated by a blank space:
    For example, axiom list `ClassA1⊏ClassB1, ClassA2⊏ClassB2` will be saved as:
    ```
    ClassA1 ClassB1
    ClassA2 ClassB2
    ```
    
    @params
    axioms: list of tuples (str, str): axiom list to save
    dirname: str, directory path
    """
    with open(File.AXIOMS.path(dirname), "w") as f:
        for child, parent in axioms:
            print(child, parent, file=f)
            
def load_dataset(dirname):
    elements = dict()
    elements["axioms"] = load_axioms(dirname)
    elements["name2cls"], elements["cls2name"] = load_classes(dirname)
    elements["indices"], elements["labels"] = load_instances(dirname)
    return elements    
            
def save_dataset(data, dirname):
    if os.path.exists(dirname):
        x = input(f"{dirname} already exists. Are you sure you want to overwrite it? y/[n]")
        if x != "y": return
    os.makedirs(dirname, exist_ok=True)
    
    axiom_file = os.path.join(dirname, "axioms")
    clname_file = os.path.join(dirname, "name_to_index")

    save_axioms(data.axioms, dirname)
    save_classes(data.name2cls, dirname)
    save_instances(data.indices, data.labels, dirname)