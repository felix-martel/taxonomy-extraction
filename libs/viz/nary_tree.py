import matplotlib.pyplot as plt
from libs.tree import Node
from libs.utils.misc import namer

def __demo_tree():
    t = Node("a")
    t.add_many("bcd")
    t["b"].add_many("efg")
    t["c"].add("h")
    t["e"].add_many("ij")
    return t


def get_coords(tree, step_x=1, step_y=-0.2):
    coords = dict()
    edges = []
    def get_coords(node, offset=0):
        dx = step_x / 2**node.depth
        y = node.depth * step_y
        if node.is_leaf:
            mi = offset
            x = offset + dx / 2
            ma = offset + dx
        else:
            X, Y = [], []
            mi = offset
            offset = mi
            ma = None
            for child in node.children:
                xc, yc, mic, mac = get_coords(child, offset=offset)
                X.append(xc)
                offset = mac
                ma = mac
            x = (ma + mi) / 2
        coords[node] = (x, y, mi, ma)
        for child in node.children:
            (xa, ya, *_), (xb, yb, *_) = coords[child], coords[node]
            edges.append(([xa, xb], [ya, yb]))
        return (x, y, mi, ma)
    get_coords(tree)
    return {node: (x, y) for node, (x, y, *_) in coords.items()}, edges


def plot_tree(coords, edges, labels=None, edge_params=None, node_params=None, label_params=None):
    if edge_params is None:
        edge_params = dict(c="k", alpha=0.2)
    if node_params is None:
        node_params = dict(c="k", alpha=0.2)
    if label_params is None:
        label_params = dict()

    X, Y = zip(*coords.values())
    plt.scatter(X, Y, **node_params)
    for edge in edges:
        plt.plot(*edge, **edge_params)

    if labels is not None:
        get_name = namer(labels)

        for node, (x, y) in coords.items():
            plt.annotate(get_name(node), (x, y), **label_params)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    t = __demo_tree()
    coords, edges = get_coords(t)
    plot_tree(coords, edges, labels="name")
