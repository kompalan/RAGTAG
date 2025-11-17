from typing import Optional, Any
import ast
from git import Repo
import shutil
import json

class FunctionDef:
    def __init__(self, name: str, source: str):
        self.name = name
        self.source = source

class ModuleDependencies(ast.NodeVisitor):
    def __init__ (self, module_name: str, filepath: str):
        self.deps = {}
        self.filepath = filepath
        self.toplevel_name = module_name

        self.source = ""
        with open(self.filepath, "r") as f:
            self.source = f.read()
        
    def get_dep_graph (self):
        return self.deps

    def visit_Import(self, node: ast.Import) -> Any:
        # Straightforward import statement eg. import module
        return super().generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        # from module import something
        return super().generic_visit(node)
    
    def visit_alias(self, node: ast.alias) -> Any:
        # import module as alias
        return super().generic_visit(node)
    
    def visit_TypeAlias(self, node: ast.TypeAlias) -> Any:
        return super().generic_visit(node)
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        # print(f"Function Def: {ast.dump(node)}\n")
        try:
            decorator_names = '\n'.join([f"@{ast.unparse(deco)}" for deco in node.decorator_list])
            seg = ast.get_source_segment(self.source, node)
            function_src = f"{decorator_names}\n{seg}"
            
            print(function_src)
            print('\n')
        except Exception as e:
            print(f"Could not get source segment: {e}\n")

        # This should register a new node in the graph if not present
        if node.name not in self.deps:
            self.deps[f"{self.toplevel_name}.{node.name}"] = []

        return super().generic_visit(node)
    
    def visit_Call(self, node: ast.Call) -> Any:
        # This should add an edge between function and node
        # print(f"Function Call: {ast.dump(node)}\n")

        func_name = None
        module_name = None

        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            module_name = None
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
            if isinstance(node.func.value, ast.Name):
                module_name = node.func.value.id

        if func_name:
            # Assuming the last defined function is the caller
            if self.deps:
                last_func = list(self.deps.keys())[-1]
                self.deps[last_func].append(f"{module_name}.{func_name}" if module_name else func_name)
            
        return super().generic_visit(node)
    

def make_graph_from_src (root_path: str):
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
    
    module_subgraphs = []
    for py_file in py_files:
        print(f"Processing file: {py_file}")
        with open(py_file, "r") as f:
            source_code = f.read()
            tree = ast.parse(source_code)
            module_name = py_file.stem
            visitor = ModuleDependencies(module_name=module_name, filepath=str(py_file))
            visitor.visit(tree)
            dep_graph = visitor.get_dep_graph()
            module_subgraphs.append((py_file, module_name, dep_graph))
            
            if dep_graph:
                break
    
    print('\n\n')
    for py_file, module_name, subgraph in module_subgraphs:
        print(f"Dependency graph for {py_file} (module: {module_name}):")
        print(json.dumps(subgraph, indent=4))

        if subgraph:
            break # For demo purposes, only print the first nonempty one.

    return py_files

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