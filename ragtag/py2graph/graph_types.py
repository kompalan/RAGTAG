from enum import Enum
from typing import Any, Optional
from uuid import uuid4


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
        self.src = ""


class FunctionNode(Node):
    def __init__(self, name: Any, node: Any, function_src: Optional[str]):
        super().__init__(name, node, NodeType.FUNCTION)
        self.src = function_src
