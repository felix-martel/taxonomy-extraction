"""
Contains `Node`, the base class for representing trees.
Basic tree ideas: a Node is identified to its subtree, that is a Tree can be:
- a leaf Tree (no children)
- a Tree containing subtrees

# TODO: add an 'export as HTML' feature
# TODO: add a 'plot with Plotly' feature
# TODO: add static creation methods 'from_tree', 'from_revtree', + dump/load methods
# TODO: add the notion of *tree compression* (subtree containing only selected nodes)
"""
from typing import List, Union, Dict, Tuple, NewType, Iterable, Callable, Generator, TextIO

from collections import deque
from contextlib import redirect_stdout
from io import StringIO

from libs.tree.pprint import print_tree

INF = float("inf")

NodeId = NewType("NodeId", str)
NodeOrName = Union[str, NodeId, "Node"]
Number = Union[int, float]


class Node:
    """
    Base class for representing a tree and a node a in a tree (same thing)
    """
    def __init__(self,
                 name: str,
                 parent: Union[None, "Node"] = None,
                 tree: Union[None, Dict[NodeId, "Node"]] = None) -> None:
        if (tree is not None and name in tree) or (parent is not None and name in parent.tree):
            raise ValueError(f"Node({name}) already exists. Node names must be unique")
        self.name: NodeId = NodeId(name)
        self.parent: Union[None, "Node"] = parent
        self.children: List["Node"] = []
        self.depth: int = 0 if parent is None else parent.depth+1
        if tree is None:
            tree = {} if parent is None else parent.tree
        self.tree: Dict[NodeId, "Node"] = tree
        self.tree[self.name] = self

        if parent:
            self.parent.children.append(self)

    def add(self, name: str) -> "Node":
        """
        Create a new node with name `name`, and add it to the children of `self`
        """
        node = Node(name, parent=self, tree=self.tree)
        return node

    def remove(self, name_or_node: NodeOrName) -> None:
        """
        Remove a node from the children of `self`. You can select the node to remove whether by its name or by itself
        """
        node = self[name_or_node]
        if node in self.children:
            node.detach()
        del node  # probably useless

    def add_many(self, names: Iterable[str]) -> None:
        """
        Create and add a bunch of new nodes from a list, see method `Node.add` for details
        """
        for name in names:
            self.add(name)

    def remove_many(self, names: Iterable[str]) -> None:
        """
        Remove a bunch of nodes from `self.children`, see method `Node.remove` for details
        """
        for name in names:
            self.remove(name)

    def siblings(self) -> List["Node"]:
        """
        Return the siblings of the node (i.e all other nodes that have the same parent)
        """
        if self.is_root:
            return []
        return [node for node in self.parent.children if node != self]

    def to_edges(self) -> List[Tuple[NodeId, NodeId]]:
        """
        Return a list of `(child, parent)` tuples, each one representing an edge in the tree
        """
        edges = [(node.name, node.parent.name) for node in self if node.parent is not None]
        return edges

    @classmethod
    def from_edges(cls, edges: List[Tuple[str, str]], add_root: Union[bool, str, NodeId] = False) -> "Node":
        """
        Build a tree from a list of `(child, parent)` tuples, each one representing an edge in the tree
        """
        children, parents = zip(*edges)
        nodes = {*children, *parents}
        roots = set(parents) - set(children)
        if len(roots) < 1:
            raise ValueError("Tree contains a cycle")
        elif len(roots) > 1:
            if not add_root:
                raise ValueError("Tree contains several roots. You must set a value for parameter 'add_root'")
            for root in roots:
                edges.append((root, add_root))
            roots = {add_root}
        root = roots.pop()
        tree = cls(root)
        node_dict = {node: cls(node) for node in nodes}
        node_dict[root] = tree

        for child, parent in edges:
            child, parent = node_dict[child], node_dict[parent]
            child.attach(parent)
        tree.rebuild_depths()
        return tree

    @classmethod
    def read_edge_list(cls, file: TextIO) -> List[Tuple[str, str]]:
        edges = []
        for line in file:
            child, parent = line.split()
            edges.append((child, parent))
        return edges

    @classmethod
    def write_edge_list(cls, edges: Iterable[Tuple[str, str]], file: TextIO):
        for edge in edges:
            print(*edge, file=file)

    @classmethod
    def from_file(cls, filename: str, add_root: Union[bool, str, NodeId] = False) -> "Node":
        with open(filename, "r", encoding="utf8") as f:
            edges = cls.read_edge_list(f)
        return cls.from_edges(edges, add_root=add_root)

    def to_file(self, filename: str):
        with open(filename, "w", encoding="utf8") as f:
            self.write_edge_list(self.to_edges(), f)

    @property
    def is_root(self) -> bool:
        return self.parent is None

    @property
    def is_leaf(self) -> bool:
        return not bool(self.children)

    @property
    def n_children(self) -> int:
        return len(self.children)

    @property
    def root(self) -> "Node":
        return self

    def __iter__(self) -> Generator["Node", None, None]:
        """
        Return all nodes in the tree, including itself (i.e non-strict successors of `self`)
        """
        for node in self.tree.values():
            if node <= self:
                yield node

    def iter_children(self) -> Generator["Node", None, None]:
        """
        Return all nodes in the tree, excluding itself (i.e strict successors of `self`)
        """
        for node in self.tree.values():
            if node < self:
                yield node

    def rebuild_depths(self, from_depth: Union[bool, int] = None) -> None:
        """
        Recompute depths for all successors of `self`
        """
        if from_depth is not None:
            self.depth = from_depth
        for node in self.dfs():
            if node.parent is not None:
                node.depth = node.parent.depth + 1

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name})"

    def __str__(self) -> str:
        return str(self.name)

    def __lt__(self, other) -> bool:
        """
        Return True iff `self` is a strict successor of `other` (that is, `self` belong to the subtree `other`)
        """
        if not isinstance(other, type(self)) or (self.parent is None):
            return False
        if self.parent == other:
            return True
        return self.parent < other

    def __le__(self, other) -> bool:
        """
        Return True iff `self` is a non-strict successor of `other` (that is, `self` belong to the subtree `other`)
        """
        return self == other or self < other

    def __hash__(self):
        # Name are supposed to be unique in the tree
        return hash(self.name)

    def __contains__(self, item) -> bool:
        """
        # TODO: what's the diff with `self < other`?
        """
        if isinstance(item, type(self)):
            item = item.name
        return item in self.tree and self.tree[item] <= self

    def __getitem__(self, item) -> "Node":
        """
        Get a node in the tree from its name
        """
        if self.tree is None:
            raise ValueError(f"{repr(self)}.tree is not set: can't access item with syntax 'node[key]'")

        node = item if isinstance(item, type(self)) \
            else self.tree[item]
        if node <= self:
            return node
        raise KeyError(f"Can't find item '{item}' in the subtree of {repr(self)}")

    def bfs(self, start: Union[None, NodeOrName] = None, max_depth: Number = INF, max_nodes: Number = INF,
            halting_func: Union[None, Callable] = None, **kwargs) -> Generator["Node", None, None]:
        """Breadth-first search over the tree"""
        if start is None:
            start = self.root
        else:
            start = self[start]
        max_depth += start.depth
        curr_nodes = 0
        unvisited = deque([start])
        while unvisited and curr_nodes < max_nodes:
            node = unvisited.popleft()
            curr_nodes += 1
            yield node
            if node.is_leaf or node.depth >= max_depth or (halting_func is not None and halting_func(node, **kwargs)):
                continue
            for children in node.children:
                unvisited.append(children)

    def dfs(self, start: Union[None, NodeOrName] = None, max_depth: Number = INF, max_nodes: Number = INF,
            halting_func: Union[None, Callable] = None, **kwargs) -> Generator["Node", None, None]:
        """Depth-first search over the tree"""
        if start is None:
            start = self.root
        else:
            start = self[start]
        curr_nodes = 0
        unvisited = [start]
        while unvisited and curr_nodes < max_nodes:
            node = unvisited.pop()
            curr_nodes += 1
            yield node
            if node.is_leaf or node.depth >= max_depth or (halting_func is not None and halting_func(node, **kwargs)):
                continue
            for children in node.children:
                unvisited.append(children)

    def detach(self, base_node: Union[None, NodeOrName] = None, update_tree: bool = True) -> None:
        """
        Remove the edge between `self` and its parent. Alternatively, you can detach `base_start` instead of `self`
        """
        if base_node is None:
            base_node = self
        base_node.parent.children.remove(base_node)
        base_node.parent = None
        base_node.depth = 0

        if update_tree:
            to_remove = {*base_node}
            new_tree = dict()
            for node in to_remove:
                del base_node.tree[node.name]
                new_tree[node.name] = node
            base_node.tree = new_tree
        base_node.rebuild_depths()

    def attach(self, parent: "Node", update_tree: bool = True) -> None:
        """
        Add an edge between `self` and `parent`.
        """
        if not self.is_root:
            self.detach()
        if update_tree:
            for node in self:
                parent.tree[node.name] = node
                node.tree = parent.tree
        parent.children.append(self)
        self.parent = parent
        self.rebuild_depths(from_depth=parent.depth+1)

    def move(self, to: "Node") -> None:
        """
        Move an node to the node `to` (i.e remove the current edge to its parent and add an edge to `to`)
        """
        self.detach(update_tree=False)
        self.attach(to, update_tree=False)

    def iter_branch(self) -> Generator["Node", None, None]:
        """
        Iterate along the unique path between `self` and the root
        """
        yield self
        if self.is_root:
            return
        else:
            yield from self.parent.iter_branch()

    def print(self, start=None, max_depth=5, as_string=False, **kwargs):
        """
        Provide a (text-based) visual representation of the tree.

        By default, the tree is printed, but you can set `as_string=True` to return it as a string instead. By default,
        only the 5 first levels are plotted. That behavior can be controlled with the parameter `max_depth`. You can
        also print from a different node than the root with parameter `start`.

        This is a wrapper around function `libs.tree.pprint.print_tree()`, which is itself a fork of
        `https://github.com/clemtoy/pptree`. Credits to Clément Michard.
        """
        if start is None:
            start = self
        else:
            start = self[start]
        if as_string:
            with StringIO() as writer, redirect_stdout(writer):
                    print_tree(start, max_depth=max_depth, **kwargs)
                    return writer.getvalue()
        print_tree(start, max_depth=max_depth, **kwargs)

    def custom_print(self, func, start=None, max_depth=5, **kwargs):
        """
        Define a custom printing function for the node. See method `Node.print` for the other parameters.
        """
        self.print(start, max_depth, nameattr=func, **kwargs)


a = Node.from_edges([("f", "c"), ("e", "b"), ("d", "b")], add_root="a")
b, c, d, e, f = (a[x] for x in "bcdef")
