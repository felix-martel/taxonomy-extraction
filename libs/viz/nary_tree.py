"""
Plot trees using Matplotlib.

# TODO: add support for Plotly

Example
```
t = Node("a")
t.add_many("bcd")
t["b"].add_many("efg")
t["c"].add("h")
t["e"].add_many("ij")

coords, edges = get_coords(t)
plot_tree(coords, edges, labels="name")
```
"""
import matplotlib.pyplot as plt
from libs.utils.misc import namer
from typing import Dict, Tuple, Any, List, Union
from libs.tree import Node


CoordDict = Dict[Node, Tuple[float, float]]
Edge = Tuple[Tuple[float, float], Tuple[float, float]]
EdgeList = List[Edge]
PlotParams = Union[None, Dict[str, Any]]


def get_coords(tree: Node, step_x: float = 1., step_y: float = -0.2) -> Tuple[CoordDict, EdgeList]:
    """
    Find coordinates of each nodes in the tree
    """
    coords = dict()
    edges: EdgeList = []

    def rec_get_coords(node: Node, offset: float = 0.) -> Tuple[float, float, float, float]:
        dx = step_x / 2**node.depth
        y = node.depth * step_y
        if node.is_leaf:
            mi = offset
            x = offset + dx / 2
            ma = offset + dx
        else:
            xs, ys = [], []
            mi = offset
            offset = mi
            ma = None
            for child in node.children:
                xc, yc, mic, mac = rec_get_coords(child, offset=offset)
                xs.append(xc)
                offset = mac
                ma = mac
            x = (ma + mi) / 2
        coords[node] = (x, y, mi, ma)
        for child in node.children:
            (xa, ya, *_), (xb, yb, *_) = coords[child], coords[node]
            edges.append(((xa, xb), (ya, yb)))
        return x, y, mi, ma
    rec_get_coords(tree)
    return {node: (x, y) for node, (x, y, *_) in coords.items()}, edges


def plot_tree(coords: CoordDict, edges: Edge, labels=None, filename: Union[None, str] = None,
              edge_params: PlotParams = None, node_params: PlotParams = None, label_params: PlotParams = None) -> None:
    """
    Plot a tree from a list of coordinates and edges.

    :param coords: mapping from nodes to (x, y) coordinate
    :param edges: list of edges (edges are represented as a tuple [x1, x2], [y1, y2]
    :param labels: how to label the nodes. If None, nodes won't have labels. See `libs.misc.namer` for details
    :param filename: if not None, the figure will be saved to `filename`
    :param edge_params: parameters for displaying the edges (passed to `plt.plot`)
    :param node_params: parameters for displaying the nodes (passed to `plt.scatter`)
    :param label_params: parametres for displaying the labels (passed to `plt.annotate`)
    :return:
    """
    if edge_params is None:
        edge_params = dict(c="k", alpha=0.2)
    if node_params is None:
        node_params = dict(c="k", alpha=0.2)
    if label_params is None:
        label_params = dict()

    plt.scatter(*zip(*coords.values()), **node_params)
    for edge in edges:
        plt.plot(*edge, **edge_params)

    if labels is not None:
        get_name = namer(labels)

        for node, (x, y) in coords.items():
            plt.annotate(get_name(node), (x, y), **label_params)
    plt.axis("off")
    if filename is not None:
        plt.savefig(fname=filename)
    plt.show()
