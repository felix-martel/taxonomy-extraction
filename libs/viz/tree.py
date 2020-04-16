"""
Display interactive tree using Plotly.

How to use:
```
>>> from libs.clustering import Clustering
>>> clu = Clustering(...).fit(...)
>>> G = create_graph(clu)
>>> # Add extra info (for node label, node color or hovertext) as a dict(node->dict(key->info)) where keys can be set using
arguments label_key, hover_key, color_key
>>> node_info = {node: {"name": node.name, "precision": node.precision()} for node in G.nodes()}
>>> fig = to_plotly(G, node_info, color_key="precision")
>>> fig.show()
```
Two main methods:
`create_graph` to initialize the graph from a clustering tree. Return a networkx.DiGraph object.
`to_plotly` to create the figure containing the graph, node information, and more

Three default dicts of parameters (can be useful to see the available options):
LABEL_PARAMS: passed to the 'annotation' argument of method 'go.Layout'
EDGE_PARAMS: passed to the 'go.Scatter' method used to create edges
NODE_PARAMS:  passed to the 'marker' argument of method 'go.Scatter'  used to create nodes
LAYOUT_PARAMS: passed to the 'go.Layout' method
"""
import networkx as nx
import plotly.graph_objs as go

def create_graph(clu, start=None, size_threshold=20, max_depth=5, compress=True):
    """
    clu: 'libs.Clustering' object
    start: 'libs.Cluster' or int, cluster to use as root. If None, use 'clu.root'
    size_threshold: int, if a cluster has less than 'size_threshold' elements, it won't be plotted
    max_depth: int, clusters deeper than 'max_depth' won't be plotted
    compress: bool, whether to plot or not clusters with only one child
    """
    def sibling(c):
        if c.parent is None: return set()
        return (set(c.parent.children) - {c}).pop()

    G = nx.DiGraph()
    to_compress = dict()
    nodes = set()
    if start is not None: max_depth += start.depth
    for c in clu.bfs(start=start, max_depth=max_depth):
        nodes.add(c)
        if c == start or c.is_root or c.size < size_threshold: continue

        parent = to_compress.get(c.parent, c.parent)
        if compress:
            sib = sibling(c)
            if not sib or sib.size < size_threshold:
                to_compress[c] = parent
                continue
        G.add_edge(parent, c)
    setattr(G, "root", clu.root)
    return G #, nodes

def create_nary_graph(tree):
    G = nx.DiGraph()
    nodes = set()
    
    

def node_pos_getter(G):
    """
    Utility function to get the coordinates of G's nodes
    Return a function node --> (x, y) coordinates
    """
    _node_pos = dict()
    root = G.root
    offset_depth = root.depth

    def node_pos(c, dx=1, dy=1):
        if c in _node_pos:
            return _node_pos[c]
        else:
            if c == root: #.is_root: 
                coords = 0, 0
            else:
                x, y = node_pos(c.parent)
                sibling = (set(c.parent.children) - {c}).pop()
                if sibling not in G:
                    pos = 0
                else:
                    pos = -1 if c.parent.children[0] == c else 1
                coords = (x+pos*dx/(2**(c.depth-offset_depth)), y-dy)
            _node_pos[c] = coords
        return coords
    return node_pos

LABEL_PARAMS = dict(textangle=-30)
EDGE_PARAMS = dict(line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
NODE_PARAMS = dict(showscale=True, colorscale='YlGnBu', reversescale=True, color=[], size=10, line_width=2,
            colorbar=dict(thickness=15, title='Cluster Precision', xanchor='left', titleside='right')
        )
LAYOUT_PARAMS = dict(title='<br>Clustering Tree',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )

def to_plotly(G, node_info=None, 
              label_params=LABEL_PARAMS, 
              edge_params=EDGE_PARAMS, 
              node_params=NODE_PARAMS, 
              layout_params=LAYOUT_PARAMS,
              label_key="name", 
              hover_key=None, 
              color_key=None
             ):
    """
    Create a go.Figure object from the tree G. 
    Use 'node_info' alongside arguments 'label_key', 'hover_key' and 'color_key' to add names, texts on hover and colors to each node
    """
    node_pos = node_pos_getter(G)
    
    # Edges
    edge_x = []
    edge_y = []
    for a, b in G.edges():
        x0, y0 = node_pos(a)
        x1, y1 = node_pos(b)

        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(x=edge_x, y=edge_y, **edge_params)
    
    # Nodes
    node_x, node_y, node_hover, node_label = [], [], [], []
    for node in G.nodes():
        x, y = node_pos(node)
        node_x.append(x)
        node_y.append(y)
        if node_info:
            try:
                name = node_info[node].get(label_key, "")
                node_label.append(dict(x=x, y=y, text=name, visible=bool(name), **label_params))
                if hover_key:
                    node_hover.append(node_info[node][hover_key])
            except KeyError as e:
                print(node.name, node.id)
                raise e
        else:
            node_hover.append(node.name)

    params = dict(mode="markers", marker=node_params)
    if hover_key:
        params["hovertext"] = node_hover
    node_trace = go.Scatter(x=node_x,  y=node_y, **params)
    if node_info and color_key:
        node_trace.marker.color = [node_info[c][color_key] for c in G.nodes()]

    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(annotations=node_label, **layout_params))
    return fig

