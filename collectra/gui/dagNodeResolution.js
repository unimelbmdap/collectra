// Resolves a pipeline DAG label to its real per-page instance
// id(s) via pageNodeIds (label -> real ids). Idempotent on an already-real
// id: pageNodeIds is keyed by label, so it just won't match and passes
// through unchanged.
function resolveDagClickTarget(labelOrId, pageNodeIds) {
    const ids = pageNodeIds.get(labelOrId);
    if (!ids || ids.length === 0) return { nodeId: labelOrId, instances: [labelOrId] };
    if (ids.length === 1) return { nodeId: ids[0], instances: ids };
    return { nodeId: null, instances: ids };
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { resolveDagClickTarget };
}
