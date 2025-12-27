from __future__ import annotations
from typing import Optional, Any, Dict, List, Set, Tuple
import ast
from git import Repo
import shutil
from enum import Enum
import networkx as nx
import itertools
import logging
from ragtag.py2graph.lsp_client import PyrightLsp
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4
from collections import defaultdict

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=f"{__name__}.log",
    format="%(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)


class GlobalContext:
    def __init__(self):
        self.lsp = None


class NodeType(Enum):
    MODULE = 0
    FUNCTION = 1
    CLASS = 2


class EdgeType(Enum):
    DEFINES = 0  # Modules defining classes/functions
    OWNS = 1  # Classes defining functions/attributes
    IMPORTS = 2  # Modules/Classes/Functions importing modules
    CALLS = 3  # Functions calling other functions
    INHERITS_FROM = 4  # Classes inheriting from other classes


class Node:
    def __init__(self, name: Any, node: Any, type: NodeType):
        self.name = name
        self.node = node
        self.type = type
        self.uuid = uuid4()

    #
    # def __str__(self):
    #     return f"Node:\n\tName: {self.name}\n\tType: {str(self.type)}\n\tUUID: {str(self.uuid)}"
    #


class FunctionNode(Node):
    def __init__(self, name: Any, node: Any, function_src: Optional[str]):
        super().__init__(name, node, NodeType.FUNCTION)
        self.function_src = function_src


class ModuleDependencies(ast.NodeVisitor):

    @dataclass
    class UnresolvedCall:
        func_name: str
        start_line: int
        end_line: int
        callsite: Tuple[int, int]

        def __hash__(self):
            return hash((self.func_name, self.start_line, self.end_line, self.callsite))

    def __init__(self, module_name: str, filepath: str, lsp: PyrightLsp):
        self.subgraph: nx.DiGraph = nx.DiGraph()

        self.deps: Dict = {}
        self.filepath: str = filepath
        self.toplevel_name: str = module_name
        self.filepos_to_nodes: Dict[int, List[Node]] = defaultdict(list)

        self.class_stack: List[Node] = []
        self.function_stack: List[Node] = []

        self.defined_functions: Dict[ModuleDependencies.UnresolvedCall, Node] = {}
        self.defined_classes: Dict[str, Node] = dict()
        self.defined_bases: Dict[str, List[Node]] = dict()
        self.imports: Set[Node] = set()

        # Call to function from function
        self.unresolved_calls: Dict[ModuleDependencies.FunctionTuple, Node] = {}
        self.aliases: Dict[str, str] = {}

        self.lsp = lsp

        self.source: List[str] = []
        with open(self.filepath, "r") as f:
            self.source = f.read().splitlines()

        self.module = Node(module_name, None, NodeType.MODULE)
        self.subgraph.add_node(self.module)

    def get_subgraph(self) -> nx.DiGraph:
        return self.subgraph

    def visit_Import(self, node: ast.Import) -> Any:
        logger.info(f"Visiting Import Node: {ast.dump(node, indent=4)}")

        # Straightforward import statement eg. import module
        for name in node.names:
            import_node = Node(name, node, NodeType.MODULE)
            self.imports.add(import_node)

        super().generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        logger.info(f"Visiting ImportFrom Node: {ast.dump(node, indent=4)}")

        # from module import something
        for name in node.names:
            import_node = Node(name, node, NodeType.MODULE)
            self.imports.add(import_node)
            for alias in import_node.node.names:
                self.aliases[alias.name] = import_node.node.module

        super().generic_visit(node)

    def visit_alias(self, node: ast.alias) -> Any:
        logger.info(f"Visiting Alias Node: {ast.dump(node, indent=4)}")

        # import module as alias
        self.imports.add(Node(node.name, node, NodeType.MODULE))
        super().generic_visit(node)

    def visit_TypeAlias(self, node: Any) -> Any:
        return super().generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        logger.info(f"Visiting ClassDef Node: {ast.dump(node, indent=4)}")

        class_node = Node(node.name, node, NodeType.CLASS)
        self.defined_classes[node.name] = class_node

        # TODO: Find a better way to add a constructor. Maybe add a function call when __init__ is defined?
        # self.defined_functions[(self.toplevel_name, node.name, "A")] = FunctionNode(
        #     node.name, node, None
        # )
        #
        self.filepos_to_nodes[node.lineno].append(class_node)
        self.subgraph.add_node(class_node)
        self.subgraph.add_edge(self.module, class_node, edge_type=EdgeType.DEFINES)

        for base in node.bases:
            self.defined_bases[node.name] = node.bases
        self.class_stack.append(class_node)
        super().generic_visit(node)
        self.class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        logger.info(f"Visiting FunctionDef Node: {ast.dump(node, indent=4)}")

        def get_function_source_segment(node: ast.FunctionDef):
            function_src = None
            try:
                decorator_names = "\n".join(
                    [f"@{ast.unparse(deco)}" for deco in node.decorator_list]
                )
                seg = ast.get_source_segment("\n".join(self.source), node)
                function_src = f"{decorator_names}\n{seg}"
            except Exception as e:
                logger.warning(f"Could not get source segment: {e}\n")

            return function_src

        function_src = get_function_source_segment(node)
        parent_class = None
        function_name = node.name
        fully_qualified_name = function_name
        logger.info(f"Found: {fully_qualified_name}")
        if len(self.class_stack) > 0:
            parent_class = self.class_stack[-1].name
            fully_qualified_name = f"{self.toplevel_name}.{
                parent_class}.{function_name}"
        else:
            fully_qualified_name = f"{self.toplevel_name}.{function_name}"

        function_node = FunctionNode(fully_qualified_name, node, function_src)
        self.filepos_to_nodes[node.lineno].append(function_node)
        self.subgraph.add_node(function_node)
        self.defined_functions[(self.toplevel_name, parent_class, function_name)] = (
            function_node
        )

        self.function_stack.append(function_node)
        # Add an edge between child and parent
        if parent_class:
            self.subgraph.add_edge(
                self.class_stack[-1], function_node, edge_type=EdgeType.OWNS
            )
        else:
            self.subgraph.add_edge(
                self.module, function_node, edge_type=EdgeType.DEFINES
            )

        super().generic_visit(node)
        self.function_stack.pop()

    def _get_callsite_position(
        self, start_line: int, code_lines: List[str], symbol: str
    ):
        def py_col_to_lsp_col(line: str, col: int) -> int:
            return len(line[:col].encode("utf-16-le")) // 2

        for i, line in enumerate(code_lines):
            pos = line.find(symbol)
            if pos != -1:
                return (start_line + i + 1, py_col_to_lsp_col(line, pos))
        else:
            return (None, None)

    def visit_Call(self, node: ast.Call) -> Any:
        logger.info(f"Visiting Call Node: {ast.dump(node, indent=4)}")

        # This should add an edge between function and node
        func_name = None
        class_name = None
        module_name = None

        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
            if isinstance(node.func.value, ast.Name):
                module_name = node.func.value.id
            elif isinstance(node.func.value, ast.Call):
                # If we're in a class, the function call could be special, like super()
                # This gets complicated fast, especially because we don't have MRO information here.
                # Instead, we concede that super() can refer to any of the base classes.
                # TODO: Mostly we're going to rely on the LSP to deal with this. But we might need
                # to have some heuristic in case the LSP can't find the definition
                pass

        function_tuple = (module_name, class_name, func_name)
        parent_function_or_module = (
            self.function_stack[-1] if len(self.function_stack) > 0 else self.module
        )

        if tuple((module_name, class_name, func_name)) not in self.defined_functions:
            source_dump = "\n".join(
                [f"{i+1}\t{seg}" for i, seg in enumerate(self.source)]
            )

            segment = "\n".join(self.source[node.lineno - 1 : node.end_lineno])

            position = (
                node.lineno - 1,
                node.end_lineno,
                self._get_callsite_position(
                    node.lineno - 1,
                    self.source[node.lineno - 1 : node.end_lineno],
                    func_name,
                ),
            )

            logger.warning(
                f"Couldn't resolve {func_name}. Here's the function I found {position}:\n{segment}\nFull Source:\n === START SOURCE ===\n{source_dump}\n==== END SOURCE ===\n"
            )
            self.unresolved_calls[
                ModuleDependencies.UnresolvedCall(
                    func_name, position[0], position[1], position[2]
                )
            ] = parent_function_or_module
        else:
            # Get me the function def for this call, and connect it to the latest called
            # function
            logger.info(
                f"Resolved Function Call. {parent_function_or_module} calls: {func_name}"
            )
            self.subgraph.add_edge(
                parent_function_or_module,
                self.defined_functions[function_tuple],
                edge_type=EdgeType.CALLS,
            )

        return super().generic_visit(node)


class Subgraph:
    def __init__(self, graph: nx.DiGraph, lsp: PyrightLsp, visitor: ModuleDependencies):
        self.graph = graph
        self.visitor = visitor
        self.lsp = lsp

    def _link_imports(
        self,
        src_path: str,
        other_subgraph: Subgraph,
        merged_graph: nx.DiGraph,
        subgraphs: Dict[str, Subgraph],
    ):
        for import_node in self.visitor.imports:
            import_name = None
            if isinstance(import_node.node, ast.ImportFrom) or isinstance(
                import_node.node, ast.Import
            ):
                import_name = import_node.node.module
            else:
                continue

            logger.debug(f"Found Import: {import_name}\n\t{ast.dump(import_node.node)}")

            if other_subgraph.visitor.toplevel_name == import_name:
                # Link the module nodes
                merged_graph.add_edge(
                    self.visitor.module,
                    other_subgraph.visitor.module,
                    edge_type=EdgeType.IMPORTS,
                )

        return merged_graph

    def _link_functions(
        self,
        src_path: str,
        other_subgraph: Subgraph,
        merged_graph: nx.DiGraph,
        subgraphs: Dict[str, Subgraph],
    ):
        logger.info(f"Starting LSP Server with Source Path: {src_path}")
        for unresolved_call, function_node in self.visitor.unresolved_calls.items():
            file_path = self.visitor.filepath
            row, column = unresolved_call.callsite
            self.lsp.open_file(file_path)
            logger.info(
                f"Attempting to find definition for {unresolved_call.func_name} ({self.visitor.filepath}:{(row, column)})"
            )
            symbol_definition = self.lsp.definition(file_path, row - 1, column)

            if symbol_definition:
                logger.info(f"Found definition!\n{symbol_definition}")
                for location in symbol_definition:
                    if location.uri in subgraphs:
                        logger.info(f"Found Subgraph for URI: {location.uri}")
                        linked_subgraph = subgraphs[location.uri]

                        logger.info(
                            f"Looking for {unresolved_call.func_name} in linked visitor"
                        )

                        print(
                            {
                                i: [node.name for node in nodes]
                                for i, nodes in linked_subgraph.visitor.filepos_to_nodes.items()
                            }
                        )
                        if (
                            location.range.start.line + 1
                            in linked_subgraph.visitor.filepos_to_nodes
                        ):
                            logger.info(
                                f"Found Nodes on Line {location.range.start.line + 1}"
                            )
                            for node in linked_subgraph.visitor.filepos_to_nodes[
                                location.range.start.line + 1
                            ]:
                                logger.info(f"Searching Node: {node.name}")
                                if unresolved_call.func_name in node.name:
                                    logger.info(f"Found matching Node: {node}")
                                    merged_graph.add_edge(
                                        function_node, node, edge_type=EdgeType.CALLS
                                    )
            else:
                logger.warning(
                    f"Did not find a definition for {unresolved_call.func_name}. Details of the query: {(file_path, row, column)}"
                )
            # for (
            #     mod_name,
            #     class_name,
            #     func_name,
            # ), function_node in self.visitor.unresolved_calls.items():
            #     logger.debug(
            #         f"Unresolved Call: {function_node.node.name} -> {mod_name},{class_name},{func_name}"
            #     )
            #     for (
            #         mod_name2,
            #         class_name2,
            #         func_name2,
            #     ), other_function_node in other_subgraph.visitor.defined_functions.items():
            #         if (
            #             (func_name == func_name2)
            #             and (mod_name == mod_name2)
            #             and (class_name == class_name2)
            #         ):
            #             # Link the function nodes
            #             merged_graph.add_edge(
            #                 function_node,
            #                 other_function_node,
            #                 edge_type=EdgeType.CALLS,
            #             )
            #
        return merged_graph

    def _link_classes(
        self,
        src_path: str,
        other_subgraph: Subgraph,
        merged_graph: nx.DiGraph,
        subgraphs: Dict[str, Subgraph],
    ):
        return merged_graph

    def link(
        self,
        src_path: str,
        other_subgraph: Subgraph,
        merged_graph: nx.DiGraph,
        subgraphs: Dict[str, Subgraph],
    ):
        merged_graph = self._link_imports(
            src_path, other_subgraph, merged_graph, subgraphs
        )
        merged_graph = self._link_functions(
            src_path, other_subgraph, merged_graph, subgraphs
        )
        merged_graph = self._link_classes(
            src_path, other_subgraph, merged_graph, subgraphs
        )
        return merged_graph


def save_graph_picture_to_file(
    graph: nx.DiGraph,
    filepath: str,
    size: Tuple[int, int] = (40, 40),
    dpi: int = 500,
    font_size: int = 2,
):
    """Save a picture of the graph to a file."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=size, dpi=dpi)
    pos = nx.nx_pydot.graphviz_layout(graph)
    # Draw nodes and labels separately and draw edges with arrows for directed graphs
    nx.draw_networkx_nodes(graph, pos, node_color="lightblue")

    label_data = {node: node.name for node in graph}
    edge_data = {(u, v): d["edge_type"].name for u, v, d in graph.edges(data=True)}
    nx.draw_networkx_labels(
        graph, pos, labels=label_data, font_size=font_size, font_weight="bold"
    )
    nx.draw_networkx_edges(graph, pos, arrows=True)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_data, font_size=font_size)
    plt.axis("off")
    plt.savefig(filepath)
    plt.close()


def make_subgraphs(
    root_path: str, lsp: PyrightLsp
) -> List[Tuple[nx.DiGraph, ModuleDependencies]]:
    """Given a file path to a root directory, traverse subdirectories
    for python files and make a graph of function dependencies.
    """
    # Rglob traverse for python files.
    logger.info(f"Got: {root_path}")

    def find_python_files(path: str):
        from pathlib import Path

        p = Path(path)
        return list(p.rglob("*.py"))

    py_files = find_python_files(root_path)
    logger.info(f"Found {len(py_files)} python files.")

    subgraphs: Dict[str, Subgraph] = {}
    for py_file in py_files:
        logger.info(f"Processing file: {py_file}")

        # logger.info(f"Sending didOpen for: {py_file}")
        # lsp.open_file(py_file)
        with open(py_file, "r") as f:
            source_code = f.read()
            tree = ast.parse(source_code)
            module_name = py_file.stem
            visitor = ModuleDependencies(
                module_name=module_name, filepath=str(py_file), lsp=lsp
            )
            visitor.visit(tree)

            subgraphs[(Path(py_file).resolve()).as_uri()] = Subgraph(
                visitor.get_subgraph(), lsp, visitor
            )

    return subgraphs


def link_subgraphs(
    subgraphs: Dict[str, Subgraph], src_path: str, lsp: PyrightLsp
) -> nx.DiGraph:
    """Given a list of subgraphs, link them together based on imports."""
    dep_graph = nx.DiGraph()
    for uri, subgraph in subgraphs.items():
        dep_graph = nx.compose(dep_graph, subgraph.graph)

    # For each unique pair of subgraphs, link functions, imports and classes
    for subgraph_a, subgraph_b in itertools.combinations(subgraphs.values(), 2):
        dep_graph = subgraph_a.link(src_path, subgraph_b, dep_graph, subgraphs)

    return dep_graph


@contextmanager
def make_lsp(workspace_root: str = "."):
    lsp = PyrightLsp(workspace_root=workspace_root)

    try:
        lsp.start()
        yield lsp
    finally:
        lsp.stop()


def make_graph_from_src(src_path: str):
    """Given a file path to a root directory, traverse subdirectories
    for python files and make a graph of function dependencies.
    """
    from pathlib import Path

    with make_lsp(src_path) as lsp:
        root_path = str(Path(src_path).resolve())
        subgraphs = make_subgraphs(root_path, lsp)
        dep_graph = link_subgraphs(subgraphs, root_path, lsp)
    return dep_graph


def make_graph_from_github(
    repo_name: str,
    repo_url: Optional[str],
    commit_hash: Optional[str],
    target_path: str = "/tmp",
):
    """Given a repo name and an optional commit hash,
    clone repo and checkout to commit hash before traversing.
    """
    graph: Optional[nx.DiGraph] = None
    if repo_url:
        path = f"{target_path}/{repo_name}"

        shutil.rmtree(path, ignore_errors=True)

        # Clone repo from github to path
        repo = Repo.clone_from(repo_url, f"{target_path}/{repo_name}")

        if repo.working_tree_dir:
            # Checkout to commit_hash if applicable
            if commit_hash:
                repo.git.checkout(commit_hash)

            # Call make_graph_from_src on the root path.
            graph = make_graph_from_src(str(repo.working_tree_dir))

    return graph
