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

// What clicking a DAG node opens: its label's tab, on the label's first
// instance. A node with no instance on the bound page does nothing (null).
// Before any page is bound there is nothing to check against, so the label
// is opened as-is.
function dagNodeOpenTarget(label, pageNodeIds, pageBound) {
    if (!pageBound) return { groupKey: label, nodeId: label };
    const ids = pageNodeIds.get(label);
    if (!ids || ids.length === 0) return null;
    return { groupKey: label, nodeId: ids[0] };
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { resolveDagClickTarget, dagNodeOpenTarget };
}
