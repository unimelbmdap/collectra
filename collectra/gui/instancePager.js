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

if (typeof module !== 'undefined') module.exports = {pagerState, blankTabMessage};
