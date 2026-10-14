const {tabMenuItems} = require('../collectra/gui/tabContextMenu.js');

const nodeTab = {id: 't1', kind: 'node', nodeId: 'primary_label_unoriented1', groupKey: 'primary_label_unoriented'};

it('offers Delete for the instance shown, then Close tab, on a node tab', () => {
    expect(tabMenuItems(nodeTab, true)).toEqual([
        {action: 'delete', label: 'Delete "primary_label_unoriented1"'},
        {action: 'close', label: 'Close tab'},
    ]);
});

it('omits Close tab when the tab cannot be closed', () => {
    expect(tabMenuItems(nodeTab, false).map((i) => i.action)).toEqual(['delete']);
});

it('only offers Close tab on a DAG tab, never Delete', () => {
    const dag = {id: 'd', kind: 'dag', nodeId: null};
    expect(tabMenuItems(dag, true)).toEqual([{action: 'close', label: 'Close tab'}]);
    expect(tabMenuItems(dag, false)).toEqual([]);
});

it('offers nothing for a missing tab or one with no node to delete', () => {
    expect(tabMenuItems(null, true)).toEqual([]);
    expect(tabMenuItems({id: 'x', kind: 'node', nodeId: null}, false)).toEqual([]);
});

it('is wired into the page: module loaded and tabs open the menu on right-click', () => {
    const html = require('node:fs').readFileSync(require.resolve('../collectra/gui/index.html'), 'utf8');
    expect(html).toContain('<script src="tabContextMenu.js"></script>');
    expect(html).toContain(`oncontextmenu="showTabContextMenu(event, '\${panelKey}', '\${tab.id}')"`);
    expect(html).toContain('function showTabContextMenu(');
});
