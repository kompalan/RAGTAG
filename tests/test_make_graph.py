from ragtag.py2graph.make_graph import make_graph_from_src, save_graph_picture_to_file
from ragtag.py2graph.serializer import serialize, deserialize
from pathlib import Path
from networkx import is_isomorphic


def make_simple_graph():
    graph = make_graph_from_src("./tests/golden/simple_test")
    save_graph_picture_to_file(
        graph, "plots/simple_test_graph.png", size=(40, 40), dpi=300, font_size=12
    )

    # /home/anurag/latest-project/tests/serialized
    path = Path("./tests/serialized/simple_test.json").resolve()

    serialize(path, graph)

    new_graph = deserialize(path)

    save_graph_picture_to_file(
        new_graph,
        "plots/simple_test_graph_repro.png",
        size=(40, 40),
        dpi=300,
        font_size=12,
    )

    assert is_isomorphic(graph, new_graph)


make_simple_graph()
