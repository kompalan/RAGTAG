from ragtag.py2graph.make_graph import make_graph_from_src, Node, FunctionNode

def make_simple_graph():
    graph = make_graph_from_src('./tests/golden/simple_test')

make_simple_graph()