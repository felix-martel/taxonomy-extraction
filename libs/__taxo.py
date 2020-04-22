"""
DEPRECATED
"""
from collections import defaultdict
from warnings import warn

from libs.axiom import transitive_closure
from libs.dataset import Dataset
from libs.pptree import print_tree
import os
import re
from itertools import chain
from collections import Counter


warn("'libs.taxo' is deprecated. Use module 'libs.taxonomy' instead", category=DeprecationWarning)

class TaxonomyItem:
    def __init__(self, class_name, parent=None, selected=True):
        self.name = class_name
        self.parent = parent
        self.selected = selected
        self.children = []
        if self.parent is None:
            self.depth = 0
        else:
            self.depth = self.parent.depth + 1
            
    def __bool__(self):
        return self.selected
    
    def __repr__(self):
        return "Class.{}.{}".format(self.depth, self.name)
    
    def __str__(self):
        return "{}.{}".format(self.name, self.depth)
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        if not isinstance(other, TaxonomyItem):
            return False
        return (self.name == other.name and
                ((self.parent is None and other.parent is None) or 
                 (self.parent is not None and other.parent is not None and self.parent.name == other.parent.name)) and
                self.depth == other.depth and
                set(c.name for c in self.children) == set(c.name for c in other.children) and
                self.selected == other.selected)
                
    
    def __getitem__(self, index):
        if index >= self.depth or index < 0:
            return self
        else:
            return self.parent[index]
        
    @property
    def selected_children(self):
        return [child for child in self.children if isinstance(child, TaxonomyItem) and child.selected]
        
    def add_child(self, *args):
        self.children.extend(args)
        
    def set_children(self, children):
        self.children = list(children)
        
    def selected_parent(self):
        if self.parent is None:
            return self
        c = self.parent
        while not c.selected:
            c = c.parent
            if c is None:
                break
        return c
    
    @property
    def all(self):
        _p = [self.name]
        curr = self
        while curr.parent is not None:
            curr = curr.parent
            _p.append(curr.name)
        return _p
    
    @property
    def hierarchy(self):
        return "/".join(self.all)
    
    def is_a(self, cls):
        return cls in self.all
        
    def clip_at(self, depth, selected_only=False):
        """
        Return the class-equivalent of depth `depth`
        
        depth: no clipping if depth < 0
        selected_only : 
        
        e.g. we have hierarchy=SoccerPlayer/Athlete/Person/Agent/Thing, then we'll get :
        clip_at(-1): SoccerPlayer
        clip_at(4) : SoccerPlayer
        clip_at(3) : Athlete
        clip_at(2) : Person
        clip_at(1) : Agent
        clip_at(0) : Thing
        """
        if (self.selected or not selected_only) and (depth < 0 or self.depth <= depth):
            # No clipping
            return self
        else:
            return self.parent.clip_at(depth, selected_only=selected_only)
        
                 
                 
class Taxonomy:
    def __init__(self, root, children):        
        data = {}
        unprocessed = [(root, None)]
        while unprocessed:
            name, parent = unprocessed.pop()
            node = TaxonomyItem(name, parent=parent)
            if parent is not None:
                parent.add_child(node)
            data[node.name] = node
            
            for child in children[name]:
                unprocessed.append((child, node))
        # Mapping name --> TaxonomyItem
        self.data = data
        # Mapping depth --> class list
        self.at_depth = defaultdict(list)
        self.root = self.data[root]
        for item in self:
            self.at_depth[item.depth].append(item)
        
        
    def __iter__(self):
        for item in self.data.values():
            yield item
            
    def __len__(self):
        return len(self.data)
    
    def __contains__(self, key):
        return key in self.data
    
    def __getitem__(self, key):
        return self.data[key]
    
    def __eq__(self, other):
        if not isinstance(other, Taxonomy): return False
        return all(self[k] == other[k] for k in chain(self.keys(), other.keys()))
            
    def keys(self):
        yield from self.data.keys()
    
    def items(self):
        yield from self.data.items()
        
    def select(self, func, *args, **kwargs):
        for item in self:
            item.selected = func(item, *args, **kwargs)
            
    def select_from_iterable(self, i):
        i = set(i)
        for item in self:
            item.selected = item.name in i
        
    @classmethod
    def from_axioms(cls, axioms, add_root=False):
        children = defaultdict(set)
        roots = {b for _, b in axioms}
        for a, b in axioms:
            children[b].add(a)
            if a in roots:
                roots.remove(a)

        if len(roots) > 1:
            if add_root:
                root = add_root
                if not isinstance(root, str): 
                    raise TypeError(f"Root must be a str, not a {type(root)}")
                for child in roots:
                    children[root].add(child)
                roots = {root}
            else:
                raise ValueError("Taxonomy has multiple roots")
        elif len(roots) < 1:
            raise ValueError("Taxonomy contains a cycle")
        root = roots.pop()
        return cls(root, children)
    
    def to_axioms(self, closure=False):
        axioms = set()
        for a in self:
            if a.selected:
                p = a.selected_parent()
                if p is not None:
                    axioms.add((a.name, p.name))
        if closure:
            axioms = transitive_closure(axioms)
        return axioms
    
    def get_equivalent(self, cls):
        item = self[cls]
        if item.selected:
            return item.name
        else:
            return item.selected_parent().name        
    
    def convert_dataset(self, dataset):
        name2cls = {item.name:i for i, item in enumerate(self) if item.selected}
        cls2name = {i:n for n,i in name2cls.items()}
        axioms = self.to_axioms()
        new_labels = []
        for label in dataset.labels:
            old_name = dataset.cls2name[label]
            new_name = self.get_equivalent(old_name)
            new_label = name2cls[new_name]
            new_labels.append(new_label)
        return Dataset(dataset.indices, new_labels, name2cls, cls2name, axioms)
    
    def print(self, selected_only=True, start=None, **kwargs):
        if "childattr" not in kwargs:
             kwargs["childattr"] = "selected_children" if selected_only else "children"
        if start is None:
            start = self.root
        print_tree(start, **kwargs)
        
    def summary(self, n_max=5, depth=2):
        def print_level(k):
            if k is None:
                return "..."
            if k in self.at_depth:
                m = len(self.at_depth[k])            
                class_list = [cls.name for cls in self.at_depth[k][:min(m, n_max)]]
                class_list = ", ".join(class_list)
                suffix = "" if n_max > m else " and {} others".format(m-n_max)
                prefix = "Level {}: ".format(k)
                return prefix+class_list+suffix
        if len(self.at_depth) > 2*depth:
            d_max = 1 + max(self.at_depth.keys())
            to_print = [*range(depth), None, *range(d_max-depth, d_max)]
        else:
            to_print = sorted(self.at_depth.keys())
        lines = [
            "{} classes".format(len(self)),
            *(print_level(k) for k in to_print)
        ]
        print(*lines, sep="\n")
        
    def save(self, filename, sep=","):
        dirname, _ = os.path.split(filename)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(filename, "w") as f:
            print(self.root.name, file=f)
            for cls in self:
                print(cls.name, *(child.name for child in cls.children), sep=sep, file=f)

    @classmethod
    def load(cls, filename, sep=","):
        children = dict()
        with open(filename, "r") as f:
            root = next(f).rstrip()
            for line in f:
                parent, *child = line.rstrip().split(sep)
                children[parent] = child
        return cls(root, children)
    
    def get_subtree(self, root):
        """
        Return a subtaxonomy with root `root`
        """
        assert root in self
        axioms = set()
        unprocessed = {self[root]}
        while unprocessed:
            node = unprocessed.pop()
            for c in node.children:
                axioms.add((c.name, node.name))
                unprocessed.add(c)
        return Taxonomy.from_axioms(axioms)
    
    def dfs(self, start=None, max_depth=float("inf"), max_nodes=float("inf"), reverse=False):
        """Depth-first search over the taxonomy"""
        if reverse:
            search = list(self.dfs(start, max_depth, max_nodes, reverse=False))
            yield from reversed(search)
            return
        if start is None: 
            start = self.root
        else:
            start = self[start]
        curr_depth = 0
        curr_nodes = 0
        unvisited = [start]
        while unvisited and curr_nodes < max_nodes:
            node = unvisited.pop()
            curr_nodes += 1
            yield node
            if not node.children or node.depth >= max_depth:
                continue
            for children in node.children:
                unvisited.append(children)
        
class ClassMapper:
    """Mapping instance-->class"""
    def __init__(self, tax, idx2cls):
        self.tax = tax
        self.to_cls = idx2cls
        
    @classmethod
    def from_dir(cls, dirname):
        tax = Taxonomy.load(os.path.join(dirname, "true_taxonomy"))
        with open(os.path.join(dirname, "true_classes"), "r") as f:
            instance_class = dict()
            for line in f:
                index, classname = line.split()
                instance_class[int(index)] = classname
        return cls(tax, instance_class)
    
    def __getitem__(self, key):
        """Get TaxonomyItem corresponding to the most specific class of entity `key`
        
        eg. let `key` be an instance from classes Athlete, Person, Agent, Thing, then
        its most specific type is 'Athlete', and we'll have 
        self[key] = Class.3.Athlete
        self[key][-1] = self[key][3] = Class.3.Athlete (level-3 class)
        self[key][0] = Class.0.Thing (level-0 class)
        self[key][1] = Class.1.Agent (level-1 class) and so on
        """
        return self.tax[self.to_cls[key]]
    
    def composition(self, entities, level=-1):
        """
        Return the level-k composition of a list of entities
        ex:
        ```
        >> l = [A Person/Thing, B Athlete/Person/Thing, C Athlete/Person/Thing]
        >> self.composition(l)
        {Athlete:2, Person:1}
        >> self.composition(l, level=2)
        {Athlete: 2, Person: 1}
        >> self.composition(l, level=1)
        {Person: 3}
        >> self.composition(l, level=0)
        {Thing: 3}
        ```
        """
        return Counter(self[e][level].name for e in entities)
    
    def most_common(self, entities, level=-1, min_weight=0.9, max_classes=5):
        """Display the overall composition of a list of entities"""
        count = self.composition(entities, level)
        n = len(entities)
        cum_weight = 0
        n_classes = 0
        sorted_classes = count.most_common()
        main_classes = []
        for cls, weight in sorted_classes:
            cum_weight += weight/n
            main_classes.append((cls, weight/n))
            if cum_weight > min_weight or len(main_classes) >= max_classes:
                break

        if len(count) > max_classes:
            main_classes.append(("other", 1-cum_weight))

        plural = "s" if n > 1 else ""        
        print("{} item{}".format(n, plural))
        for i, (name, weight) in enumerate(main_classes):
            print("{:02d} {:25.25} {:.1f}%".format(i+1, name, 100*weight))
    
    
def build_from_graph(graph, subclass_rel="rdfs:subClassOf", root_class="owl:Thing", known_issues={"dbo:PersonalEvent": "dbo:Event"}, fix_issues=True):
    VALID_TYPE_PATTERN = re.compile("^(dbo:)|^(owl:Thing)$")
    def is_valid_type(t):
        return bool(re.match(VALID_TYPE_PATTERN, t))
    
    is_subclass_of = graph.rel.to_id(subclass_rel)
    true_axioms = {(h, t) for h, _, t in graph.find_triples(r=is_subclass_of, as_string=True) 
                   if is_valid_type(h) and is_valid_type(t)}

    print(f"{len(true_axioms)} subsumption axioms found")
    
    children, parents = zip(*true_axioms)
    roots = set(parents) - set(children) - {root_class}
    print("Extra root(s) found:", *roots)

    if all(r in known_issues for r in roots) and fix_issues:
        print(f"Rooting {', '.join(a+' to '+b for a,b in known_issues.items())}.")
        for r in roots:
            true_axioms.add((r, known_issues[r]))
    if len(roots) == 1:
        return Taxonomy.from_axioms(true_axioms)
    return Taxonomy.from_axioms(true_axioms, add_root=root_class)


def html_print(tax, html_func, **kwargs):
    lines = []
    def print_tree(current_node, childattr='children', nameattr='name', display_func=print,
                   indent='', last='updown', max_depth=None, halting_func=None, print_params={}):
        """
        nameattr:
        - if string, then the name of a node will be the value of node.{nameattr}
        - if callable, then the name of a node will be nameattr(node)
        - else, use str(node)
        """
        print_children = not ( max_depth == 0 or (halting_func is not None and halting_func(current_node)))
        if max_depth is not None:
            max_depth -= 1
        if callable(nameattr):
            name = lambda node: nameattr(node)
        elif hasattr(current_node, nameattr):
            name = lambda node: getattr(node, nameattr)
        else:
            name = lambda node: str(node)

        children = lambda node: getattr(node, childattr)
        nb_children = lambda node: sum(nb_children(child) for child in children(node)) + 1
        size_branch = {child: nb_children(child) for child in children(current_node)}
        SPACE = "&nbsp;"

        if print_children:
            """ Creation of balanced lists for "up" branch and "down" branch. """
            up = sorted(children(current_node), key=lambda node: nb_children(node))
            down = []
            while up and sum(size_branch[node] for node in down) < sum(size_branch[node] for node in up):
                down.append(up.pop())

            """ Printing of "up" branch. """
            for child in up:     
                next_last = 'up' if up.index(child) is 0 else ''
                next_indent = '{0}{1}{2}'.format(indent, SPACE if 'up' in last else '│', SPACE * len(name(current_node)))
                print_tree(child, childattr, nameattr, display_func, next_indent, next_last, max_depth, halting_func, print_params)

        """ Printing of current node. """
        if last == 'up': start_shape = '┌'
        elif last == 'down': start_shape = '└'
        elif last == 'updown': start_shape = SPACE
        else: start_shape = '├'

        if print_children:
            if up: end_shape = '┤'
            elif down: end_shape = '┐'
            else: end_shape = ''
        else:
            end_shape = ""

        lines.append('{0}{1}{2}{3}'.format(indent, start_shape, html_func(current_node, end_shape)))

        if print_children:
            """ Printing of "down" branch. """
            for child in down:
                next_last = 'down' if down.index(child) is len(down) - 1 else ''
                next_indent = '{0}{1}{2}'.format(indent, SPACE if 'down' in last else '│', SPACE * len(name(current_node)))
                print_tree(child, childattr, nameattr, display_func, next_indent, next_last, max_depth, halting_func, print_params)
    print_tree(tax.root, **kwargs)
    return lines