// Ancestors and Descendants of a pipeline node, worked out from the pipeline's own
// nodes and edges (plain data) — not from a drawn graph — so they are available for
// every node whatever panels happen to be open.

const LINEAGE_TASK_TYPES = new Set(['llm', 'detector', 'processor']);

// ids of every node of the same kind (tasks with tasks, data with data) reachable from
// nodeId, at any depth, in discovery order. direction is 'ancestors' or 'descendants'.
// Nodes of the other kind are walked through but not listed. skip(id) leaves out nodes
// (and what is only reachable through them).
function lineageNeighbors(nodeId, direction, nodes, edges, skip = () => false) {
    const byId = new Map((nodes || []).map((n) => [n.id, n]));
    if (!byId.has(nodeId)) return [];
    const next = new Map();
    for (const {source, target} of edges || []) {
        const [from, to] = direction === 'ancestors' ? [target, source] : [source, target];
        if (!next.has(from)) next.set(from, []);
        next.get(from).push(to);
    }
    const isTask = (id) => byId.has(id) && LINEAGE_TASK_TYPES.has(byId.get(id).nodeType);
    const startIsTask = isTask(nodeId);
    const visited = new Set([nodeId]);
    const queue = [nodeId];
    const found = [];
    while (queue.length) {
        const current = queue.shift();
        for (const id of next.get(current) || []) {
            if (visited.has(id) || skip(id)) continue;
            visited.add(id);
            if (isTask(id) === startIsTask) found.push(id);
            queue.push(id);
        }
    }
    return found;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {lineageNeighbors};
}
