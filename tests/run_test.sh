#! /bin/bash
rm ragtag.py2graph.make_graph.log
mkdir -p ./tests/serialized
rm -rf ./tests/ast_nodes

python -m tests.test_make_graph

# rm -rf ./tests/serialized

