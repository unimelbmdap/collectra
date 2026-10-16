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

// Actual result instances have their own parent edges. Category-level DAG
// reachability must never expand those edges to unrelated sibling instances.
function artefactLineageNeighbors(nodeId, direction, parents) {
    if (!parents?.has(nodeId)) return [];
    const children = new Map();
    if (direction === 'descendants') {
        for (const [id, parentIds] of parents) {
            for (const parent of parentIds) {
                if (!children.has(parent)) children.set(parent, []);
                children.get(parent).push(id);
            }
        }
    }
    const seen = new Set([nodeId]), queue = [nodeId], found = [];
    while (queue.length) {
        const id = queue.shift();
        for (const next of (direction === 'ancestors' ? parents.get(id) : children.get(id)) || []) {
            if (seen.has(next) || !parents.has(next)) continue;
            seen.add(next); found.push(next); queue.push(next);
        }
    }
    return found;
}

function lineageTargets(nodeId, direction, nodes, edges, nodeIds, parents, resolveLabel, skip) {
    if (!nodeId) return [];
    if (parents?.has(nodeId)) return artefactLineageNeighbors(nodeId, direction, parents);
    // Task selections and views without results still use the pipeline DAG.
    return lineageNeighbors(resolveLabel(nodeId), direction, nodes, edges, skip)
        .flatMap(label => nodeIds.get(label)?.length ? nodeIds.get(label) : [label]);
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {lineageNeighbors, artefactLineageNeighbors, lineageTargets};
}
