from typing import Optional, Any, Dict, List, Set, Tuple
import ast
from git import Repo
import shutil
import json
import abc
from enum import Enum
import networkx as nx

class GlobalContext:
    def __init__ (self):
        # Relate module names to subgraph
        self.found_modules = {}

    def add_module(self, module, value):
        self.found_modules[module] = value

class NodeType(Enum):
    MODULE   = 0
    FUNCTION = 1
    CLASS    = 2

class EdgeType(Enum):
    DEFINES = 0 # Modules defining classes/functions
    OWNS    = 1 # Classes defining functions/attributes
    IMPORTS = 2 # Modules/Classes/Functions importing modules
    CALLS   = 3 # Functions calling other functions

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
        
    def __init__ (self, module_name: str, filepath: str, context: GlobalContext):
        self.context: GlobalContext = context
        self.subgraph: nx.DiGraph   = nx.DiGraph()

        self.deps: Dict         = {}
        self.filepath: str      = filepath
        self.toplevel_name: str = module_name

        self.class_stack: List[Node]    = []
        self.function_stack: List[Node] = []

        self.defined_functions: Dict[ModuleDependencies.FunctionTuple, Node] = {}
        self.defined_classes: Dict[str, Node] = dict()
        self.imports: Set[Node] = set()

        self.unresolved_calls: Dict[ModuleDependencies.FunctionTuple, Node] = {} # Call to function from function
        self.aliases: Dict[str, str] = {}

        self.source: str = ""
        with open(self.filepath, "r") as f:
            self.source = f.read()
        
        self.module = Node(module_name, None, NodeType.MODULE)
        self.subgraph.add_node(self.module)

    def get_subgraph (self) -> nx.DiGraph:
        return self.subgraph

    def visit_Import(self, node: ast.Import) -> Any:
        # Straightforward import statement eg. import module
        for name in node.names:
            import_node = Node(name, node, NodeType.MODULE)
            self.imports.add(import_node)

        super().generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        # from module import something
        for name in node.names:
            import_node = Node(name, node, NodeType.MODULE)
            self.imports.add(import_node)

        super().generic_visit(node)
    
    def visit_alias(self, node: ast.alias) -> Any:
        # import module as alias
        self.imports.add(Node(node.name, node, NodeType.MODULE))
        super().generic_visit(node)
    
    def visit_TypeAlias(self, node: Any) -> Any:
        return super().generic_visit(node)
    
    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        class_node = Node(node.name, node, NodeType.CLASS)
        self.defined_classes[node.name] = class_node
        self.subgraph.add_node(class_node)
        self.subgraph.add_edge(self.module, class_node, edge_type=EdgeType.DEFINES)
        
        self.class_stack.append(class_node)
        super().generic_visit(node)
        self.class_stack.pop()
    
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        def get_function_source_segment(node: ast.FunctionDef):
            function_src = None
            try:
                decorator_names = '\n'.join([f"@{ast.unparse(deco)}" for deco in node.decorator_list])
                seg = ast.get_source_segment(self.source, node)
                function_src = f"{decorator_names}\n{seg}"
            except Exception as e:
                print(f"Could not get source segment: {e}\n")

            return function_src

        function_src = get_function_source_segment(node)
        parent_class = None
        function_name = node.name
        fully_qualified_name = function_name
        print(f"Found: {fully_qualified_name}")
        if len(self.class_stack) > 0:
            parent_class = self.class_stack[-1].name
            fully_qualified_name = f"{self.toplevel_name}.{parent_class}.{function_name}"
        else:
            fully_qualified_name = f"{self.toplevel_name}.{function_name}"

        function_node = FunctionNode(fully_qualified_name, node, function_src)
        self.subgraph.add_node(function_node)
        self.defined_functions[(self.toplevel_name, parent_class, function_name)] = function_node
        self.function_stack.append(function_node)
        # Add an edge between child and parent
        if parent_class:
            self.subgraph.add_edge(self.class_stack[-1], function_node, edge_type=EdgeType.OWNS)
        else:
            self.subgraph.add_edge(self.module, function_node, edge_type=EdgeType.DEFINES)

        super().generic_visit(node)
        self.function_stack.pop()
    
    def visit_Call(self, node: ast.Call) -> Any:
        # This should add an edge between function and node
        func_name = None
        class_name = None
        module_name = self.toplevel_name

        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            # module_name = None
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
            if isinstance(node.func.value, ast.Name):
                module_name = node.func.value.id
        
        function_tuple = (module_name, class_name, func_name)
        parent_function_or_module = self.function_stack[-1] if len(self.function_stack) > 0 else self.module
        if (module_name, class_name, func_name) not in self.defined_functions:
            print(f"Couldn't resolve {func_name}!")
            self.unresolved_calls[(module_name, class_name, func_name)] = parent_function_or_module
        else:
            # Get me the function def for this call, and connect it to the latest called
            # function
            print(f"Yay! Resolved {func_name}")
            self.subgraph.add_edge(parent_function_or_module, self.defined_functions[function_tuple], edge_type=EdgeType.CALLS)

        return super().generic_visit(node)

def save_graph_picture_to_file(graph: nx.DiGraph, filepath: str, size: Tuple[int, int]=(40, 40), dpi: int=500, font_size: int=2):
    """Save a picture of the graph to a file.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=size, dpi=dpi)
    pos = nx.spring_layout(graph)
    # Draw nodes and labels separately and draw edges with arrows for directed graphs
    nx.draw_networkx_nodes(graph, pos, node_size=500, node_color="lightblue")

    label_data = {node:node.name for node in graph}
    edge_data = {(u, v): d['edge_type'].name for u, v, d in graph.edges(data=True)}
    nx.draw_networkx_labels(graph, pos, labels=label_data, font_size=font_size, font_weight="bold")
    nx.draw_networkx_edges(graph, pos, arrows=True)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_data, font_size=font_size)
    plt.axis('off')
    plt.savefig(filepath)
    plt.close()


def make_subgraphs(root_path: str) -> List[Tuple[nx.DiGraph, ModuleDependencies]]:
    """Given a file path to a root directory, traverse subdirectories
    for python files and make a graph of function dependencies.
    """
    # Rglob traverse for python files.
    print(f"Got: {root_path}")
    def find_python_files (path: str):
        from pathlib import Path
        p = Path(path)
        return list(p.rglob("*.py"))
    
    py_files = find_python_files(root_path)
    print(f"Found {len(py_files)} python files.")
    
    subgraphs: List[Tuple[nx.DiGraph, ModuleDependencies]] = []
    for py_file in py_files:
        print(f"Processing file: {py_file}")

        context: GlobalContext = GlobalContext()
        with open(py_file, "r") as f:
            source_code = f.read()
            tree = ast.parse(source_code)
            module_name = py_file.stem
            visitor = ModuleDependencies(module_name=module_name, filepath=str(py_file), context=context)
            visitor.visit(tree)

            context.add_module(py_file, (py_file, module_name, visitor))
            subgraphs.append((visitor.get_subgraph(), visitor))

    return subgraphs

def link_subgraphs (subgraphs: List[Tuple[nx.DiGraph, ModuleDependencies]]) -> nx.DiGraph:
    """Given a list of subgraphs, link them together based on imports.
    """
    dep_graph = nx.DiGraph()
    for subgraph, visitor in subgraphs:
        dep_graph = nx.compose(dep_graph, subgraph)
    
    # For each node in each subgraph, check its imports and link to other subgraphs.
    for subgraph, visitor in subgraphs:
        for import_node in visitor.imports:
            import_name = None
            if isinstance(import_node.node, ast.alias):
                import_name = import_node.name
            elif isinstance(import_node.node, Node):
                import_name = import_node.name
            else:
                continue
            
            # Check if this import_name matches any module in other subgraphs
            # TODO: Lots of cases to consider here for matching imports to modules. 
            # TODO: Will need to optimize the triple nested loop here.
            for other_subgraph, other_visitor in subgraphs:
                if other_visitor.toplevel_name == import_name:
                    # Link the module nodes
                    dep_graph.add_edge(visitor.module, other_visitor.module, edge_type=EdgeType.IMPORTS)

                for (mod_name, class_name, func_name), function_node in visitor.unresolved_calls.items():
                    for (mod_name2, class_name2, func_name2), other_function_node in other_visitor.defined_functions.items():
                        if (func_name == func_name2) and (mod_name == mod_name2) and (class_name == class_name2):
                            # Link the function nodes
                            dep_graph.add_edge(function_node, other_function_node, edge_type=EdgeType.CALLS)
    
    return dep_graph

def make_graph_from_src (src_path: str):
    """Given a file path to a root directory, traverse subdirectories
    for python files and make a graph of function dependencies.
    """
    from pathlib import Path
    root_path = str(Path(src_path).resolve())
    subgraphs = make_subgraphs(root_path)
    dep_graph = link_subgraphs(subgraphs)
    return dep_graph

def make_graph_from_github (repo_name: str, repo_url: Optional[str], commit_hash: Optional[str], target_path: str = "/tmp"):
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
            if (commit_hash):
                repo.git.checkout(commit_hash)

            # Call make_graph_from_src on the root path.
            graph = make_graph_from_src(str(repo.working_tree_dir))

    return graph