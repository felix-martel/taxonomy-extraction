import json
import os

RESOURCE_FILE = "resources.json"


def _get_registry(file=RESOURCE_FILE):
    """
    Read the graph registry. By default, it is located in "./resources.json" with the "graph" key.

    :param file: file containing the registry. It should be a JSON file with a "graph" key.
    :return: a dictionary mapping graph names to their location on the disk.
    """
    if not os.path.exists(file):
        return {}
    with open(file, "r") as f:
        r = json.load(f)
    return r

def _set_registry(r, file=RESOURCE_FILE):
    with open(file, "w") as f:
        json.dump(r, f, indent=4)

def get_graph_registry(file=RESOURCE_FILE):
    """
    Read the graph registry. By default, it is located in "./resources.json" with the "graph" key.

    :param file: file containing the registry. It should be a JSON file with a "graph" key.
    :return: a dictionary mapping graph names to their location on the disk.
    """
    if not os.path.exists(file):
        return {}
    with open(file, "r") as f:
        r = json.load(f)
    return r.get("graph", {})


def registry_contains(graph_name):
    return graph_name in get_graph_registry()


def get_registered(graph_name):
    return get_graph_registry().get(graph_name, graph_name)


def register(graph_name, graph_dir, force_override=False):
    r = _get_registry()
    if "graph" not in r:
        r["graph"] = {}
    if graph_name in r["graph"] and not force_override:
        if input(f"A graph named '{graph_name}' is already registered, with path '{r['graph'][graph_name]}'. Do you"
                 " want to override it? y/[n]") != "y":
            return
    r["graph"][graph_name] = graph_dir
    _set_registry(r)


def deregister(graph_name):
    r = _get_registry()
    if "graph" not in r or graph_name not in r["graph"]:
        return
    del r["graph"][graph_name]
    _set_registry(r)
