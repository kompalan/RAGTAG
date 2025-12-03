# RAGTAG: Retrieval-Augmented Generation on Text-Attributed Graphs

RAGTAG is a work-in-progress framework for understanding large codebases by leveraging their inherent structure. Instead of treating a repository as a bag of independent chunks, RAGTAG encodes the codebase as a _text-attributed dependency-graph_. This means that each function, class and module is attributed with an embedding of its semantics, and linked to one another as a graph. A picture of the graph created by RAGTAG is shown at the bottom of this README. The long-term goal is to investigate whether explicitly representing data in this way can improve accuracy over just using text.

RAGTAG is still in early-stage development, and any PRs and feedback would be greatly appreciated!

## Feature Roadmap

### Graph Construction

RAGTAG parses a repository and produces a dependency-graph:

* Nodes represent functions, classes, or modules
* Edges capture dependency relations (call edges, import edges, etc.)
* Each node stores text attributes: source code, docstring, comments, etc.

Graph building is implemented in `py2graph/make_graph.py` and can be extended to AST-level or file-level graphs.


### Graph-Based Retrieval (WIP)

A message-passing GNN encodes each node and predicts a **relevance score** given a natural language query. Early prototype uses:

* Feature extractor from node text
* GNN layers to propagate structural information
* A linear head producing a retrieval score

This enables retrieval that is *structure-aware*, not just text-similarity-driven.

---

### Full RAG Pipeline

Given a query:

1. Embed query
2. Rank nodes in the graph
3. Retrieve a structurally-coherent subgraph
4. Feed this subgraph into an LLM to solve downstream tasks

<img width="1000" height="1000" alt="simple_test_graph" src="https://github.com/user-attachments/assets/b4e31b0b-61ce-4b3a-8470-49d0410002a6" />
