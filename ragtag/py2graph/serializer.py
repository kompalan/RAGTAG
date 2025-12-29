from networkx import DiGraph
from pathlib import Path
import json

from ragtag.py2graph.graph_types import NodeType, EdgeType, Node, FunctionNode
import os
import pickle


def serialize(filepath: str, graph: DiGraph) -> None:
    serialized = {"schema": "codegraphv0.1", "directed": True, "nodes": [], "edges": []}
    nodes = []
    path = Path(filepath).resolve()
    for node, attrs in graph.nodes(data=True):
        nodes.append(node.node)

        serialized_node = {
            "uuid": str(node.uuid),
            "name": node.name,
            "type": node.type.name,
            "node": f"{str(node.uuid)}.pkl",
            "src": node.src,
        }

        serialized["nodes"].append(serialized_node)

    for u, v, attrs in graph.edges(data=True):
        edge = {
            "start": str(u.uuid),
            "end": str(v.uuid),
            "type": attrs["edge_type"].name,
        }
        serialized["edges"].append(edge)

    parent = path.parent.resolve() / "ast_nodes"
    os.mkdir(parent)
    for node, s_node in zip(nodes, serialized["nodes"]):
        child = parent / s_node["node"]
        with open(child, "wb+") as fp:
            pickle.dump(node, fp)

    with open(path, "w+") as fp:
        json.dump(serialized, fp)


def deserialize(filepath: str) -> DiGraph:
    description = {}
    path = Path(filepath).resolve()
    ast_nodes = Path(path.parent).resolve()
    graph = DiGraph()
    uuid_to_digraph_node = {}

    with open(filepath, "r") as fp:
        description = json.load(fp)

    assert (
        description["schema"] == "codegraphv0.1"
    ), f"Schema {description['schema']} not supported"

    for node in description["nodes"]:
        ast_fp = Path(ast_nodes / "ast_nodes" / node["node"]).resolve()

        pickled_node = None
        with open(ast_fp, "rb") as fp:
            pickled_node = pickle.load(fp)

        digraph_node = Node(node["name"], pickled_node, NodeType[node["type"].upper()])
        uuid_to_digraph_node[node["uuid"]] = digraph_node
        graph.add_node(digraph_node)

    for edge in description["edges"]:
        graph.add_edge(
            uuid_to_digraph_node[edge["start"]],
            uuid_to_digraph_node[edge["end"]],
            edge_type=EdgeType[edge["type"].upper()],
        )

    return graph
