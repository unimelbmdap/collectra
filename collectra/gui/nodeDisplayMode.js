// Pure decision logic for what clicking a DAG node should do — extracted out of
// selectNode() so it can be unit tested without a real DOM/Cytoscape/pywebview
// bridge. Loaded as a plain classic script in index.html (defines a global
// function, same as everything else there), and via require() in tests/.
//
// realType must already be resolved by the caller as pageNodeTypes.get(nodeId)
// falling back to the DAG's own `detail` field — this function doesn't know
// about either of those, it just decides given the resolved facts.
function decideNodeDisplayMode({ nodeId, realType, pageViewerActive }) {
    const idLower = (nodeId || '').toLowerCase();
    const type = realType || '';

    // Exact match only — a substring check would also catch task nodes like
    // "tei_converter" or "markdown_converter", wrongly opening the TEI/MD
    // file editor for a node that isn't the file-backed text node at all.
    const isFileBacked = idLower === 'tei' || idLower === 'markdown';
    if (isFileBacked) {
        return 'fileBackedText';
    }

    const isAnnotationText = type.includes('Text');
    if (isAnnotationText) {
        return 'annotationText';
    }

    if (pageViewerActive) {
        return 'imageHighlight';
    }

    return 'legacyNodeDisplay';
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { decideNodeDisplayMode };
}