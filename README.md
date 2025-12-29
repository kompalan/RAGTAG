# RAGTAG

RAGTAG is an experiment to test whether explicit structure helps LLM reliability relative to just text in correctness-sensitive areas, like coding.

RAGTAG is still in early-stage development, and any PRs and feedback would be greatly appreciated!

## Feature Roadmap

### Graph Construction

RAGTAG parses a repository and produces a dependency-graph:

* Nodes represent functions, classes, or modules
* Edges capture dependency relations (call edges, import edges, etc.)
* Each node stores text attributes: source code, docstring, comments, etc.

Graph building is implemented in `py2graph/make_graph.py` and can be extended to AST-level or file-level graphs.


### Graph-Based Retrieval (WIP)

A message-passing GNN encodes each node and predicts a relevance score given a natural language query. Early prototype uses:

* Feature extractor from node text
* GNN layers to propagate structural information
* A linear head producing a retrieval score

This enables retrieval that is _structure-aware_, not just text-similarity-driven.

---

### Full RAG Pipeline

Given a query:

1. Embed query
2. Rank nodes in the graph
3. Retrieve a structurally-coherent subgraph
4. Feed this subgraph into an LLM to solve downstream tasks

<img width="1000" height="1000" alt="simple_test_graph" src="https://github.com/user-attachments/assets/b4e31b0b-61ce-4b3a-8470-49d0410002a6" />
