// Options for the create-box modal's crop-type dropdown.
//
// adjacency: Map<label, label[]> of the pipeline DAG's forward edges.
// typeOf(label) -> the node's resolved type string (pipeline type, with the
// page's real results.yaml type as a fallback for nodes the pipeline leaves
// untyped).

// The only crop type string in any results.yaml is "collectra.ImageCrop";
// matches the backend's own "ImageCrop" in node_type convention.
function isImageCropType(type) {
    return (type || '').includes('ImageCrop');
}

// ImageCrop-type labels one data-generation downstream of focusedLabel —
// excludes focusedLabel itself and, unlike a plain adjacency walk, a crop's
// own further descendants (so a downstream re-crop doesn't leak into an
// ancestor's options).
//
// The pipeline graph always puts a task node (detect/ocr/...) between two data
// nodes, so a literal one-hop check would only ever find the task, never
// the crop it produces. isTaskNode(label) says which nodes are those
// transparent processing steps — the walk passes through them but stops
// at the first non-task node each path reaches.
function imageCropDescendantLabels(focusedLabel, adjacency, typeOf, isTaskNode) {
    const seen = new Set([focusedLabel]);
    let frontier = [focusedLabel];
    const result = [];
    while (frontier.length) {
        const next = [];
        for (const label of frontier) {
            for (const child of adjacency.get(label) || []) {
                if (seen.has(child)) continue;
                seen.add(child);
                if (isTaskNode(child)) {
                    next.push(child);
                } else if (isImageCropType(typeOf(child))) {
                    result.push(child);
                }
            }
        }
        frontier = next;
    }
    return result;
}

// First label with no instance on the current page (pageNodeIds: label ->
// real id list), else the first label, else "".
function firstUnusedLabel(labels, pageNodeIds) {
    if (labels.length === 0) return '';
    const unused = labels.find((l) => {
        const ids = pageNodeIds.get(l);
        return !ids || ids.length === 0;
    });
    return unused || labels[0];
}

// specific_epithet_image -> "Specific Epithet Image"
function humanizeLabel(label) {
    if (!label) return '';
    return label
        .split(/[_\s]+/)
        .filter(Boolean)
        .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
        .join(' ');
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { isImageCropType, imageCropDescendantLabels, firstUnusedLabel, humanizeLabel };
}
