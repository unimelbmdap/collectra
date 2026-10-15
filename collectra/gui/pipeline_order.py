"""A reading order for the pipeline's DAG labels.

The DAG is drawn in layers from the top down, and left to right within a
layer. This gives each label a rank that follows that order, so lists of
matches can be walked the way the pipeline reads.
"""


def pipeline_label_ranks(graph: dict) -> dict[str, int]:
    """Rank every node of a ``gui_graph()`` dictionary, 0 first.

    A node's layer is the longest chain of parents above it. Within a layer,
    nodes keep the order in which the edges first reach them (and, for nodes
    with no parent, their order in the node list).
    """
    ids = [node["id"] for node in graph.get("nodes", [])]
    edges = [(e["source"], e["target"]) for e in graph.get("edges", [])]
    for source, target in edges:
        for node_id in (source, target):
            if node_id not in ids:
                ids.append(node_id)
    node_index = {node_id: i for i, node_id in enumerate(ids)}

    parents = {node_id: [] for node_id in ids}
    first_reached: dict[str, int] = {}
    for position, (source, target) in enumerate(edges):
        parents[target].append(source)
        first_reached.setdefault(target, position)

    depth: dict[str, int] = {}

    def layer(node_id: str, path: frozenset) -> int:
        if node_id in depth:
            return depth[node_id]
        above = [p for p in parents[node_id] if p not in path]  # ignore cycles
        depth[node_id] = (
            1 + max(layer(p, path | {node_id}) for p in above) if above else 0
        )
        return depth[node_id]

    for node_id in ids:
        layer(node_id, frozenset())

    ordered = sorted(
        ids,
        key=lambda n: (depth[n], first_reached.get(n, -1), node_index[n]),
    )
    return {node_id: rank for rank, node_id in enumerate(ordered)}
