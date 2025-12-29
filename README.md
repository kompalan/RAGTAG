# RAGTAG

## What is this?

RAGTAG is an experimental retrieval framework that tests whether explicit structural information improves LLM reliability over text-only retrieval, especially in correctness-sensitive tasks like code understanding and generation.

Rather than asking a model to infer structure implicitly from raw code, RAGTAG makes structure explicit and evaluates how that changes downstream behavior.

## Why Explicit Structure?

Large language models operate under strict context and attention limits. This creates a tradeoff: you can include more raw code, or you can include higher-level information that biases the model toward the right parts of the codebase.

In principle, an LLM can infer structure from code alone—but doing so requires repeatedly rediscovering relationships that are already known at retrieval time. RAGTAG shifts this work earlier in the pipeline by explicitly encoding dependency structure in a compact, text-based form. The result is a stronger prior at generation time, with less reliance on inference-time pattern discovery.

The hypothesis is simple: trading prompt space for explicit structure improves reliability, particularly for cross-file reasoning, dependency awareness, and precise code localization.

> This project is early-stage and exploratory. Feedback, discussion, and PRs are welcome.

## Feature Roadmap
### Graph Construction

RAGTAG parses a repository and builds a dependency graph:

- Nodes represent functions, classes, or modules

- Edges capture dependency relationships (e.g. calls, imports)

Each node stores text attributes such as source code, docstrings, and comments

Initial graph construction lives in `py2graph/make_graph.py`. The design is intentionally extensible and can be adapted to finer-grained AST-level graphs or coarser file-level graphs.

### Text-Based Retrieval with Explicit Structure (WIP)

The codebase graph is serialized into a flat, text-only representation that explicitly encodes structural relationships (e.g. containment, dependency, and call edges). This serialization is injected into the prompt of a tool-calling LLM alongside retrieved code snippets.

We evaluate whether supplying explicit structural priors—rather than relying on the model to infer them implicitly—improves code-generation performance, especially on tasks that require correct localization and multi-file reasoning.

### Soft-Token Retrieval with Explicit Structure (Planned)

Flat text descriptions are simple but expensive in context space. As a next step, RAGTAG explores whether soft tokens / prefix tokens can act as a compression mechanism for these precomputed structural descriptions.

The idea is to embed structural summaries and pass them to the model via learned prefix or soft tokens, reducing prompt length while preserving structural bias. Performance will be compared against text-based structure under fixed context budgets.

<!--
---
### Full RAG Pipeline

Given a query:

1. Embed query
2. Rank nodes in the graph
3. Retrieve a structurally-coherent subgraph
4. Feed this subgraph into an LLM to solve downstream tasks

<img width="1000" height="1000" alt="simple_test_graph" src="https://github.com/user-attachments/assets/b4e31b0b-61ce-4b3a-8470-49d0410002a6" />
-->
