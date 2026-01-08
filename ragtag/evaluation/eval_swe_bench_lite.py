"""Load samples from SWE-bench_Lite, clone repos, and build a graph of .py files.

This script will:
 - load the `princeton-nlp/SWE-bench_Lite` dataset (small subset)
 - for each sample, read the `repo` and `base_commit` fields and attempt to
   clone the repository into `repos/<owner>__<name>/` (if not already present)
 - walk the repository and add nodes for Python files (*.py) and their
   parent directories to a NetworkX DiGraph, with edges directory -> file

Only `.py` files and their direct parent directories are tracked.

Usage:
    python load_codesamples.py --limit 10

Dependencies: datasets, networkx
"""

from typing import Iterable, Optional
from ..py2graph.make_graph import make_graph_from_github, save_graph_picture_to_file
from ..py2graph.serializer import serialize
import networkx as nx
from datasets import load_dataset
import os


def samples_from_dataset(limit: Optional[int] = None) -> Iterable[dict]:
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="dev")
    for i, item in enumerate(ds):
        if limit is not None and i >= limit:
            break
        yield item


seen = set()
for sample in samples_from_dataset(limit=None):
    repo = sample.get("repo", None)

    if repo:
        repo_name = repo.split("/")[-1]
        repo_url = f"https://github.com/{repo}"
        setup_commit = f"{sample.get('environment_setup_commit', None)}"

        if (repo_name, setup_commit) in seen:
            continue

        seen.add((repo_name, setup_commit))

        graph: Optional[nx.DiGraph] = make_graph_from_github(
            repo_name, repo_url, setup_commit
        )

        if graph:
            dir = f"data/{repo_name}_{setup_commit}"
            os.mkdir(dir)
            serialize(f"{dir}/graph_{repo_name}_{setup_commit}.tar.gz", graph)

        # if graph:
        # print(f"Writing {repo_name}")
        # save_graph_picture_to_file(
        #     graph, f"plots/swebench_graphs/{repo_name}_graph_{setup_commit}.png"
        # )
