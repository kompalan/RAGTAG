from ragtag.py2graph.make_graph import make_graph_from_src, save_graph_picture_to_file


def make_simple_graph():
    graph = make_graph_from_src("./tests/golden/simple_test")
    save_graph_picture_to_file(
        graph, "plots/simple_test_graph.png", size=(40, 40), dpi=300, font_size=12
    )


make_simple_graph()
