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
    
    def visit_alias(self, node: Any) -> Any:
        # import module as alias
        self.imports.add(node)
        super().generic_visit(node)
    
    def visit_TypeAlias(self, node: Any) -> Any:
        return super().generic_visit(node)
    
    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        class_node = Node(node.name, node, NodeType.CLASS)
        self.defined_classes[node.name] = class_node
        self.subgraph.add_node(class_node)
        
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
        if (module_name, class_name, func_name) not in self.defined_functions:
            print(f"Couldn't resolve {func_name}!")
            self.unresolved_calls[(module_name, class_name, func_name)] = self.function_stack[-1]
        else:
            # Get me the function def for this call, and connect it to the latest called
            # function
            print(f"Yay! Resolved {func_name}")
            self.subgraph.add_edge(self.function_stack[-1], self.defined_functions[function_tuple])

        return super().generic_visit(node)
    

def make_graph_from_src (src_path: str):
    """Given a file path to a root directory, traverse subdirectories
    for python files and make a graph of function dependencies.
    """
    from pathlib import Path
    root_path = str(Path(src_path).resolve())

    # Rglob traverse for python files.
    print(f"Got: {root_path}")
    def find_python_files (path: str):
        from pathlib import Path
        p = Path(path)
        return list(p.rglob("*.py"))
    
    py_files = find_python_files(root_path)
    print(f"Found {len(py_files)} python files.")
    
    subgraphs: List[nx.DiGraph] = []
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
            subgraphs.append(visitor.get_subgraph())

    return subgraphs

def make_graph_from_github (repo_name: str, repo_url: Optional[str], commit_hash: Optional[str], target_path: str = "/tmp"):
    """Given a repo name and an optional commit hash, 
    clone repo and checkout to commit hash before traversing.
    """
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
            py_files = make_graph_from_src(str(repo.working_tree_dir))
            return py_files
        
    return []