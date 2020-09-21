import json
import os

RESOURCE_FILE = "resources.json"
VALID_RESOURCES = ["graph", "taxonomy", "dataset"]

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


def register_resource(key, value, resource_type, file=RESOURCE_FILE, strict=True):
    r = _get_registry(file)
    if strict and resource_type not in r:
        if input(f"You're about to register the pair ('{key}', '{value}') as a custom resource '{resource_type}'. Proceed? [y]/n") == "n":
            return
    if resource_type not in r:
        r[resource_type] = dict()
    r[resource_type][key] = value
    _set_registry(r, file)


def get_registered(key, type, file=RESOURCE_FILE):
    return _get_registry(file).get(type, {}).get(key, key)
