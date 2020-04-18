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
from typing import List, Union, Dict, Tuple, Iterable, Callable, Generator, TextIO,\
    Generic, Hashable, TypeVar, Optional, Type

from collections import deque
from contextlib import redirect_stdout
from io import StringIO

import libs.viz.nary_tree as treeplot
from libs.tree.pprint import print_tree

INF = float("inf")

T = TypeVar("T", int, str, Hashable)
TNode = TypeVar("TNode", bound="Node")

Number = Union[int, float]
NodeDict = Union[Dict[T, TNode]]
NodeId = Union[T, TNode]  # A node can be identified by its name or by itself


class Node(Generic[T]):
    """
    Base class for representing a tree and a node a in a tree (same thing)
    """
    def __init__(self: TNode,
                 id_: T,
                 name: Optional[str] = None,
                 parent: Optional[TNode] = None,
                 tree: Optional[NodeDict] = None) -> None:
        if (tree is not None and id_ in tree) or (parent is not None and id_ in parent.tree):
            raise ValueError(f"Node({id_}) already exists. Node names must be unique")
        if name is None:
            name = str(id_)
        self.id: T = id_
        self.name: str = name
        self.parent = parent
        self.children: List[TNode] = []
        self.depth: int = 0 if parent is None else parent.depth+1
        if tree is None:
            tree = {} if parent is None else parent.tree
        self.tree: NodeDict = tree
        self.tree[self.id] = self

        if parent:
            self.parent.children.append(self)

    def add(self: TNode, id_: T, name: Optional[str] = None) -> TNode:
        """
        Create a new node with id `id`, and add it to the children of `self`
        """
        node = self.__class__(id_, name=name, parent=self, tree=self.tree)
        return node

    def remove(self: TNode, id_or_node: NodeId) -> None:
        """
        Remove a node from the children of `self`. You can select the node to remove whether by its id or by itself
        """
        node = self[id_or_node]
        if node in self.children:
            node.detach()
        del node  # probably useless

    def add_many(self: TNode, ids: Iterable[Union[T, Tuple[T, str]]]) -> None:
        """
        Create and add a bunch of new nodes from a list, see method `Node.add` for details
        """
        for id_ in ids:
            if isinstance(id_, tuple):
                self.add(*id_)
            else:
                self.add(id_)

    def remove_many(self, ids: Iterable[T]) -> None:
        """
        Remove a bunch of nodes from `self.children`, see method `Node.remove` for details
        """
        for id_ in ids:
            self.remove(id_)

    def siblings(self: TNode) -> List[TNode]:
        """
        Return the siblings of the node (i.e all other nodes that have the same parent)
        """
        if self.is_root:
            return []
        return [node for node in self.parent.children if node != self]

    def to_edges(self) -> List[Tuple[T, T]]:
        """
        Return a list of `(child, parent)` tuples, each one representing an edge in the tree
        """
        edges = [(node.id, node.parent.id) for node in self if node.parent is not None]
        return edges

    @classmethod
    def from_edges(cls: Type[TNode], edges: List[Tuple[T, T]], add_root: Optional[T] = None, **init_params) -> TNode:
        """
        Build a tree from a list of `(child, parent)` tuples, each one representing an edge in the tree
        """
        children, parents = zip(*edges)
        nodes = {*children, *parents}
        roots = set(parents) - set(children)
        if len(roots) < 1:
            raise ValueError("Tree contains a cycle")
        elif len(roots) > 1:
            if add_root is None:
                raise ValueError("Tree contains several roots. You must set a value for parameter 'add_root'")
            for root in roots:
                edges.append((root, add_root))
            roots = {add_root}
        root = roots.pop()
        tree = cls(root, **init_params)
        node_dict = {node: cls(node, **init_params) for node in nodes}
        node_dict[root] = tree

        for child, parent in edges:
            child, parent = node_dict[child], node_dict[parent]
            child.attach(parent)
        tree.rebuild_depths()
        return tree

    @classmethod
    def read_edge_list(cls, file: TextIO, preprocess: Optional[Callable[[str], T]] = None) -> List[Tuple[T, T]]:
        edges = []
        for line in file:
            child, parent = line.split()
            if preprocess is not None:
                child, parent = preprocess(child), preprocess(parent)
            edges.append((child, parent))
        return edges

    @classmethod
    def write_edge_list(cls, edges: Iterable[Tuple[T, T]], file: TextIO):
        for edge in edges:
            print(*edge, file=file)

    @classmethod
    def from_file(cls: Type[TNode], filename: str, add_root: Optional[T] = None,
                  preprocess: Optional[Callable[[str], T]] = None) -> TNode:
        with open(filename, "r", encoding="utf8") as file:
            edges = cls.read_edge_list(file, preprocess=preprocess)
        return cls.from_edges(edges, add_root=add_root)

    def to_file(self, filename: str):
        with open(filename, "w", encoding="utf8") as file:
            self.write_edge_list(self.to_edges(), file)

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
    def root(self: TNode) -> TNode:
        return self

    def __iter__(self: TNode) -> Generator[TNode, None, None]:
        """
        Return all nodes in the tree, including itself (i.e non-strict successors of `self`)
        """
        for node in self.tree.values():
            if node <= self:
                yield node

    def iter_children(self: TNode) -> Generator[TNode, None, None]:
        """
        Return all nodes in the tree, excluding itself (i.e strict successors of `self`)
        """
        for node in self.tree.values():
            if node < self:
                yield node

    def rebuild_depths(self, from_depth: Union[None, bool, int] = None) -> None:
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

    def __getitem__(self: TNode, item) -> TNode:
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

    def bfs(self: TNode, start: Optional[NodeId] = None, max_depth: Number = INF, max_nodes: Number = INF,
            halting_func: Optional[Callable[..., bool]] = None, **kwargs) -> Generator[TNode, None, None]:
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

    def dfs(self: TNode, start: Optional[NodeId] = None, max_depth: Number = INF, max_nodes: Number = INF,
            halting_func: Union[None, Callable[..., bool]] = None, **kwargs) -> Generator[TNode, None, None]:
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

    def detach(self, base_node: Optional[NodeId] = None, update_tree: bool = True) -> None:
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

    def attach(self: TNode, parent: TNode, update_tree: bool = True) -> None:
        """
        Add an edge between `self` and `parent`.
        """
        if not self.is_root:
            self.detach()
        if update_tree:
            for node in self:
                parent.tree[node.id] = node
                node.tree = parent.tree
        parent.children.append(self)
        self.parent = parent
        self.rebuild_depths(from_depth=parent.depth+1)

    def move(self: TNode, to: TNode) -> None:
        """
        Move an node to the node `to` (i.e remove the current edge to its parent and add an edge to `to`)
        """
        self.detach(update_tree=False)
        self.attach(to, update_tree=False)

    def iter_branch(self: TNode) -> Generator[TNode, None, None]:
        """
        Iterate along the unique path between `self` and the root
        """
        yield self
        if self.is_root:
            return
        else:
            yield from self.parent.iter_branch()

    def print(self: TNode, start: Optional[NodeId] = None, max_depth=5, as_string=False, **kwargs):
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

    def custom_print(self: TNode, func, start: Optional[NodeId] = None, max_depth=5, **kwargs):
        """
        Define a custom printing function for the node. See method `Node.print` for the other parameters.
        """
        self.print(start, max_depth, nameattr=func, **kwargs)

    def plot(self, max_depth: Optional[int] = 4, max_width: Optional[int] = 7, custom_label=None, **params):
        """
        Plot a tree using matplotlib and `libs.viz.nary_tree` module
        """
        if custom_label is None:
            custom_label = "name"
        coords, edges = treeplot.get_coords(self, max_depth=max_depth, max_width=max_width)
        treeplot.plot_tree(coords, edges, labels=custom_label, **params)

    def _build_at_depth(self: TNode) -> Dict[int, List[TNode]]:
        at_depth = dict()
        for node in self:
            if node.depth not in at_depth:
                at_depth[node.depth] = list()
            at_depth[node.depth].append(node)
        return at_depth


a = Node.from_edges([("f", "c"), ("e", "b"), ("d", "b")], add_root="a")
b, c, d, e, f = (a[x] for x in "bcdef")
