from networkx import DiGraph
from pathlib import Path
import json

from ragtag.py2graph.graph_types import NodeType, EdgeType, Node, FunctionNode
import os
import pickle
import tarfile
import uuid
import shutil


def create_tarball(tarball_name, source_dir):
    with tarfile.open(tarball_name, "w:gz") as tar:
        tar.add(source_dir, arcname=".")

def serialize(filepath: str, graph: DiGraph) -> None:
    serialized = {"schema": "codegraphv0.1", "directed": True, "nodes": [], "edges": []}
    nodes = []
    path = Path(filepath).resolve()
    parent_dir = path.parent
    temp_dir = parent_dir / f"temp_serialize_{uuid.uuid4()}"

    try:
        os.makedirs(temp_dir, exist_ok=False)
        
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

        # Write JSON to temp dir
        json_path = temp_dir / "graph.json"
        with open(json_path, "w+") as fp:
            json.dump(serialized, fp)

        # Create ast_nodes dir and write pickles
        ast_nodes_dir = temp_dir / "ast_nodes"
        os.makedirs(ast_nodes_dir, exist_ok=False)
        for node, s_node in zip(nodes, serialized["nodes"]):
            child = ast_nodes_dir / s_node["node"]
            with open(child, "wb+") as fp:
                pickle.dump(node, fp)

        # Create tarball
        create_tarball(str(path), str(temp_dir))
    
    finally:
        if temp_dir.exists():
           shutil.rmtree(temp_dir)


def deserialize(filepath: str) -> DiGraph:
    path = Path(filepath).resolve()
    parent_dir = path.parent
    temp_dir = parent_dir / f"temp_deserialize_{uuid.uuid4()}"
    
    assert path.exists(), f"Path {path} does not exist"
    
    try:
        os.makedirs(temp_dir, exist_ok=False)
        
        # Extract tarball to temp dir
        with tarfile.open(str(path), "r:gz") as tar:
            tar.extractall(path=str(temp_dir))
        
        # Read JSON from temp dir
        json_path = temp_dir / "graph.json"
        with open(json_path, "r") as fp:
            description = json.load(fp)
        
        assert (
            description["schema"] == "codegraphv0.1"
        ), f"Schema {description['schema']} not supported"
        
        graph = DiGraph()
        uuid_to_digraph_node = {}
        
        for node in description["nodes"]:
            ast_fp = temp_dir / "ast_nodes" / node["node"]
            
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
    
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
