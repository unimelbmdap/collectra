// Right-click menu entries for a tab. Delete acts on the instance the tab is
// showing, never on every instance of its label; DAG tabs have nothing to delete.
function tabMenuItems(tab, canClose) {
    if (!tab) return [];
    const items = [];
    if (tab.kind === 'node' && tab.nodeId) {
        items.push({action: 'delete', label: `Delete "${tab.nodeId}"`});
    }
    if (canClose) items.push({action: 'close', label: 'Close tab'});
    return items;
}

if (typeof module !== 'undefined') module.exports = {tabMenuItems};
