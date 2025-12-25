from __future__ import annotations
from typing import Optional, Any, Dict, List, Set, Tuple
import ast
from git import Repo
import shutil
from enum import Enum
import networkx as nx
import itertools
import logging
from lsp_client import PyrightLsp
import datetime
from contextlib import contextmanager

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=f"logs/{datetime.date.today()}/{__name__}.log",
    format="%(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)


class GlobalContext:
    def __init__(self):
        # Relate module names to subgraph
        self.found_modules = {}

    def add_module(self, module, value):
        self.found_modules[module] = value


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


class FunctionNode(Node):
    def __init__(self, name: Any, node: Any, function_src: Optional[str]):
        super().__init__(name, node, NodeType.FUNCTION)
        self.function_src = function_src


class ModuleDependencies(ast.NodeVisitor):
    FunctionTuple = Tuple[Optional[str], Optional[str], Optional[str]]

    def __init__(self, module_name: str, filepath: str, context: GlobalContext):
        self.context: GlobalContext = context
        self.subgraph: nx.DiGraph = nx.DiGraph()

        self.deps: Dict = {}
        self.filepath: str = filepath
        self.toplevel_name: str = module_name

        self.class_stack: List[Node] = []
        self.function_stack: List[Node] = []

        self.defined_functions: Dict[ModuleDependencies.FunctionTuple, Node] = {}
        self.defined_classes: Dict[str, Node] = dict()
        self.defined_bases: Dict[str, List[Node]] = dict()
        self.imports: Set[Node] = set()

        # Call to function from function
        self.unresolved_calls: Dict[ModuleDependencies.FunctionTuple, Node] = {}
        self.aliases: Dict[str, str] = {}

        self.source: str = ""
        with open(self.filepath, "r") as f:
            self.source = f.read()

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
        self.defined_functions[(self.toplevel_name, node.name, "A")] = FunctionNode(
            node.name, node
        )

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
                seg = ast.get_source_segment(self.source, node)
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

    def visit_Call(self, node: ast.Call) -> Any:
        logger.info(f"Visiting Call Node: {ast.dump(node, indent=4)}")

        # This should add an edge between function and node
        func_name = []
        class_name = []
        module_name = []

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
                if (
                    isinstance(node.func.value.func, ast.Name)
                    and node.func.value.func.id == "super"
                ):
                    for base in self.defined_bases[self.class_stack[-1]]:
                        if base in self.aliases:
                            module_name = self.aliases[base]

        function_tuple = (module_name, class_name, func_name)
        parent_function_or_module = (
            self.function_stack[-1] if len(self.function_stack) > 0 else self.module
        )
        if (module_name, class_name, func_name) not in self.defined_functions:
            logger.warning(f"Couldn't resolve {func_name}!")
            self.unresolved_calls[(module_name, class_name, func_name)] = (
                parent_function_or_module
            )
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
    def __init__(self, graph: nx.DiGraph, visitor: ModuleDependencies):
        self.graph = graph
        self.visitor = visitor

    def _link_imports(self, other_subgraph: Subgraph, merged_graph: nx.DiGraph):
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

    @contextmanager
    def lsp(workspace_root: str = "."):
        lsp = PyrightLsp(workspace_root=workspace_root)

        try:
            lsp.start()
            yield lsp
        finally:
            lsp.stop()

    def _link_functions(self, other_subgraph: Subgraph, merged_graph: nx.DiGraph):
        with self.lsp(".") as lsp:
            for (
                file_path,
                symbol,
                position,
            ), function_node in self.visitor.unresolved_calls.items():
                col, row = position
                symbol_definition = lsp.definition(file_path, row, col)
                print(symbol_definition)

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

    def _link_classes(self, other_subgraph: Subgraph, merged_graph: nx.DiGraph):
        return merged_graph

    def link(self, other_subgraph: Subgraph, merged_graph: nx.DiGraph):
        merged_graph = self._link_imports(other_subgraph, merged_graph)
        merged_graph = self._link_functions(other_subgraph, merged_graph)
        merged_graph = self._link_classes(other_subgraph, merged_graph)
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


def make_subgraphs(root_path: str) -> List[Tuple[nx.DiGraph, ModuleDependencies]]:
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

    subgraphs: List[Subgraph] = []
    for py_file in py_files:
        logger.info(f"Processing file: {py_file}")

        context: GlobalContext = GlobalContext()
        with open(py_file, "r") as f:
            source_code = f.read()
            tree = ast.parse(source_code)
            module_name = py_file.stem
            visitor = ModuleDependencies(
                module_name=module_name, filepath=str(py_file), context=context
            )
            visitor.visit(tree)

            context.add_module(py_file, (py_file, module_name, visitor))
            subgraphs.append(Subgraph(visitor.get_subgraph(), visitor))

    return subgraphs


def link_subgraphs(subgraphs: List[Subgraph], src_path: str) -> nx.DiGraph:
    """Given a list of subgraphs, link them together based on imports."""
    dep_graph = nx.DiGraph()
    for subgraph in subgraphs:
        dep_graph = nx.compose(dep_graph, subgraph.graph)

    # For each unique pair of subgraphs, link functions, imports and classes
    for subgraph_a, subgraph_b in itertools.combinations(subgraphs, 2):
        dep_graph = subgraph_a.link(subgraph_b, dep_graph)

    return dep_graph


def make_graph_from_src(src_path: str):
    """Given a file path to a root directory, traverse subdirectories
    for python files and make a graph of function dependencies.
    """
    from pathlib import Path

    root_path = str(Path(src_path).resolve())
    subgraphs = make_subgraphs(root_path)
    dep_graph = link_subgraphs(subgraphs)
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
