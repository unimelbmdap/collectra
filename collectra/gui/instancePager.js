// What the "< name >" pager above a label tab shows: the instance being
// viewed and which instances sit either side of it.
function pagerState(instances, selectedId) {
    const index = instances.indexOf(selectedId);
    const count = instances.length;
    const found = index !== -1;
    const showControls = found && count > 1;
    return {
        name: selectedId,
        count,
        position: found ? index + 1 : null,
        showControls,
        prevId: showControls && index > 0 ? instances[index - 1] : null,
        nextId: showControls && index < count - 1 ? instances[index + 1] : null,
        text: showControls ? `${index + 1} / ${count}` : '',
    };
}

// What a label tab shows when the current page has none of its node.
function blankTabMessage(tab) {
    return `This page has no ${tab.label || tab.groupKey}.`;
}

// The data type shown in a tab's hover tooltip ("collectra.Text") — not under
// the tab. DAG tabs have none; a node the page doesn't know has none either.
function tabTypeTitle(tab, pageNodeTypes, resolveLabel) {
    if (tab.kind !== 'node') return '';
    const label = tab.groupKey || (tab.nodeId ? resolveLabel(tab.nodeId) : null);
    return (label && pageNodeTypes.get(label)) || '';
}

// The buttons to the right of a node's id in the row under its tab, in order.
// Ancestors/Descendants are greyed out when the node has none.
function nodeActionButtons({hasAncestors, hasDescendants}) {
    return [
        {id: 'delete', title: 'Delete artefact', disabled: false},
        {id: 'ancestors', title: 'Ancestors', disabled: !hasAncestors},
        {id: 'descendants', title: 'Descendants', disabled: !hasDescendants},
    ];
}

if (typeof module !== 'undefined') module.exports = {pagerState, blankTabMessage, tabTypeTitle, nodeActionButtons};
