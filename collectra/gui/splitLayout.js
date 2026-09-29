// Pure decision logic for wrapping a panel with a new sibling in a given
// direction — extracted so the DOM-manipulation code (createPanel in
// index.html) only has to build the wrapper the way this says, not decide
// the shape itself.
function computeSplitLayout(direction) {
    if (direction === 'left' || direction === 'right') {
        return {
            flexDirection: 'row',
            newPanelFirst: direction === 'left',
            resizeHandleClass: 'panel-resize-handle',
        };
    }
    if (direction === 'above' || direction === 'below') {
        return {
            flexDirection: 'column',
            newPanelFirst: direction === 'above',
            resizeHandleClass: 'image-panel-resize-handle',
        };
    }
    throw new Error(`Unknown split direction: ${direction}`);
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { computeSplitLayout };
}