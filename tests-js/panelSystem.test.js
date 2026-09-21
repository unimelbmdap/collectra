const { Tab, Panel, SplitContainer, Window } = require('../collectra/gui/panelSystem.js');

describe('Tab', () => {
    it('stores id, nodeId, and label', () => {
        const tab = new Tab({ id: 't1', nodeId: 'formation', label: 'formation' });
        expect(tab.id).toBe('t1');
        expect(tab.nodeId).toBe('formation');
        expect(tab.label).toBe('formation');
    });

    it('defaults label to nodeId when not given', () => {
        const tab = new Tab({ id: 't1', nodeId: 'formation' });
        expect(tab.label).toBe('formation');
    });
});

describe('Tab kind', () => {
    it('defaults kind to "node"', () => {
        const tab = new Tab({ id: 't1', nodeId: 'formation' });
        expect(tab.kind).toBe('node');
    });

    it('accepts an explicit kind', () => {
        const tab = new Tab({ id: 't1', nodeId: null, label: 'DAG', kind: 'dag' });
        expect(tab.kind).toBe('dag');
    });
});

describe('Tab pinned', () => {
    it('defaults pinned to false', () => {
        const tab = new Tab({ id: 't1', nodeId: 'formation' });
        expect(tab.pinned).toBe(false);
    });

    it('accepts an explicit pinned flag', () => {
        const tab = new Tab({ id: 't1', nodeId: null, label: 'DAG', kind: 'dag', pinned: true });
        expect(tab.pinned).toBe(true);
    });
});

describe('Panel', () => {
    it('gets a unique auto-generated id', () => {
        const a = new Panel();
        const b = new Panel();
        expect(a.id).toBeTruthy();
        expect(a.id).not.toBe(b.id);
    });

    it('starts with no tabs and no active tab', () => {
        const panel = new Panel();
        expect(panel.tabs).toEqual([]);
        expect(panel.activeTabId).toBe(null);
        expect(panel.getActiveTab()).toBe(null);
    });

    it('addTab appends a tab and makes it active', () => {
        const panel = new Panel();
        const tab = new Tab({ id: 't1', nodeId: 'formation' });
        panel.addTab(tab);
        expect(panel.tabs).toEqual([tab]);
        expect(panel.activeTabId).toBe('t1');
        expect(panel.getActiveTab()).toBe(tab);
    });

    it('hasTab reports whether a nodeId is already open in this panel', () => {
        const panel = new Panel();
        panel.addTab(new Tab({ id: 't1', nodeId: 'formation' }));
        expect(panel.hasTab('formation')).toBe(true);
        expect(panel.hasTab('genus')).toBe(false);
    });

    it('activateTab switches which tab is active', () => {
        const panel = new Panel();
        panel.addTab(new Tab({ id: 't1', nodeId: 'formation' }));
        panel.addTab(new Tab({ id: 't2', nodeId: 'genus' }));
        panel.activateTab('t1');
        expect(panel.activeTabId).toBe('t1');
        expect(panel.getActiveTab().nodeId).toBe('formation');
    });

    it('activateTab is a no-op for an unknown tab id', () => {
        const panel = new Panel();
        panel.addTab(new Tab({ id: 't1', nodeId: 'formation' }));
        panel.activateTab('does-not-exist');
        expect(panel.activeTabId).toBe('t1');
    });

    it('closeTab removes the tab', () => {
        const panel = new Panel();
        panel.addTab(new Tab({ id: 't1', nodeId: 'formation' }));
        panel.closeTab('t1');
        expect(panel.tabs).toEqual([]);
    });

    it('closeTab falls back activeTabId to the next tab when closing the active tab', () => {
        const panel = new Panel();
        panel.addTab(new Tab({ id: 't1', nodeId: 'formation' }));
        panel.addTab(new Tab({ id: 't2', nodeId: 'genus' }));
        panel.activateTab('t1');
        panel.closeTab('t1');
        expect(panel.activeTabId).toBe('t2');
    });

    it('closeTab falls back activeTabId to the previous tab when there is no next tab', () => {
        const panel = new Panel();
        panel.addTab(new Tab({ id: 't1', nodeId: 'formation' }));
        panel.addTab(new Tab({ id: 't2', nodeId: 'genus' }));
        panel.activateTab('t2');
        panel.closeTab('t2');
        expect(panel.activeTabId).toBe('t1');
    });

    it('closeTab sets activeTabId to null when closing the only tab', () => {
        const panel = new Panel();
        panel.addTab(new Tab({ id: 't1', nodeId: 'formation' }));
        panel.closeTab('t1');
        expect(panel.activeTabId).toBe(null);
    });

    it('closeTab leaves activeTabId untouched when closing a non-active tab', () => {
        const panel = new Panel();
        panel.addTab(new Tab({ id: 't1', nodeId: 'formation' }));
        panel.addTab(new Tab({ id: 't2', nodeId: 'genus' }));
        panel.activateTab('t2');
        panel.closeTab('t1');
        expect(panel.activeTabId).toBe('t2');
    });

    it('removeTab returns the removed tab', () => {
        const panel = new Panel();
        const tab = panel.addTab(new Tab({ id: 't1', nodeId: 'formation' }));
        expect(panel.removeTab('t1')).toBe(tab);
        expect(panel.tabs).toEqual([]);
    });

    it('removeTab returns undefined and does not remove a pinned tab', () => {
        const panel = new Panel();
        const tab = panel.addTab(new Tab({ id: 't1', nodeId: null, label: 'DAG', kind: 'dag', pinned: true }));
        expect(panel.removeTab('t1')).toBe(undefined);
        expect(panel.tabs).toEqual([tab]);
    });

    it('removeTab returns undefined for an id that does not exist', () => {
        const panel = new Panel();
        expect(panel.removeTab('nope')).toBe(undefined);
    });

    it('removeTab falls back activeTabId the same way closeTab does', () => {
        const panel = new Panel();
        panel.addTab(new Tab({ id: 't1', nodeId: 'formation' }));
        panel.addTab(new Tab({ id: 't2', nodeId: 'genus' }));
        panel.activateTab('t1');
        panel.removeTab('t1');
        expect(panel.activeTabId).toBe('t2');
    });

    it('insertTab places the tab at the given index and makes it active', () => {
        const panel = new Panel();
        panel.addTab(new Tab({ id: 't1', nodeId: 'formation' }));
        panel.addTab(new Tab({ id: 't2', nodeId: 'genus' }));
        const moved = new Tab({ id: 't3', nodeId: 'age' });
        panel.insertTab(moved, 1);
        expect(panel.tabs.map((t) => t.id)).toEqual(['t1', 't3', 't2']);
        expect(panel.activeTabId).toBe('t3');
    });

    // pruneTabs is what closes stale tabs on page switch — a node that
    // was open on the old page but isn't part of the new page gets its
    // tab closed automatically instead of sitting there showing stale
    // content.

    it('pruneTabs removes tabs whose nodeId fails the predicate', () => {
        const panel = new Panel();
        panel.addTab(new Tab({ id: 't1', nodeId: 'formation' }));
        panel.addTab(new Tab({ id: 't2', nodeId: 'genus' }));
        panel.addTab(new Tab({ id: 't3', nodeId: 'age' }));
        panel.pruneTabs((nodeId) => nodeId !== 'genus');
        expect(panel.tabs.map((t) => t.nodeId)).toEqual(['formation', 'age']);
    });

    it("pruneTabs passes each tab's kind to the predicate", () => {
        const panel = new Panel();
        panel.addTab(new Tab({ id: 't1', nodeId: 'formation' }));
        panel.addTab(new Tab({ id: 't2', nodeId: null, label: 'DAG', kind: 'dag' }));
        panel.pruneTabs((nodeId, kind) => kind === 'dag');
        expect(panel.tabs.map((t) => t.id)).toEqual(['t2']);
    });

    it('pruneTabs falls back activeTabId to the first remaining tab when the active tab gets pruned', () => {
        const panel = new Panel();
        panel.addTab(new Tab({ id: 't1', nodeId: 'formation' }));
        panel.addTab(new Tab({ id: 't2', nodeId: 'genus' })); // active (last added)
        panel.pruneTabs((nodeId) => nodeId !== 'genus');
        expect(panel.activeTabId).toBe('t1');
    });

    it('pruneTabs sets activeTabId to null once every tab is pruned', () => {
        const panel = new Panel();
        panel.addTab(new Tab({ id: 't1', nodeId: 'formation' }));
        panel.pruneTabs(() => false);
        expect(panel.tabs).toEqual([]);
        expect(panel.activeTabId).toBe(null);
    });

    it('pruneTabs leaves activeTabId untouched when the active tab survives', () => {
        const panel = new Panel();
        panel.addTab(new Tab({ id: 't1', nodeId: 'formation' }));
        panel.addTab(new Tab({ id: 't2', nodeId: 'genus' }));
        panel.activateTab('t1');
        panel.pruneTabs((nodeId) => nodeId !== 'genus');
        expect(panel.activeTabId).toBe('t1');
        expect(panel.tabs.length).toBe(1);
    });

    it('openOrActivateTab creates the first tab when the panel is empty', () => {
        const panel = new Panel();
        const tab = panel.openOrActivateTab({ id: 't1', nodeId: 'formation' });
        expect(panel.tabs).toEqual([tab]);
        expect(panel.activeTabId).toBe('t1');
        expect(tab.nodeId).toBe('formation');
    });

    it('openOrActivateTab mutates the active tab in place rather than adding a new one', () => {
        const panel = new Panel();
        panel.addTab(new Tab({ id: 't1', nodeId: 'formation' }));
        const result = panel.openOrActivateTab({ id: 't2', nodeId: 'genus' });
        expect(panel.tabs.length).toBe(1);
        expect(panel.tabs[0].id).toBe('t1'); // same tab identity, content replaced
        expect(panel.tabs[0].nodeId).toBe('genus');
        expect(panel.activeTabId).toBe('t1');
        expect(result).toBe(panel.tabs[0]);
    });

    it('openOrActivateTab defaults label to nodeId when not given', () => {
        const panel = new Panel();
        panel.openOrActivateTab({ id: 't1', nodeId: 'formation' });
        expect(panel.tabs[0].label).toBe('formation');
    });

    it('openOrActivateTab adds a new tab instead of mutating a pinned active tab', () => {
        const panel = new Panel();
        panel.addTab(new Tab({ id: 'dag', nodeId: null, label: 'DAG', kind: 'dag', pinned: true }));

        const result = panel.openOrActivateTab({ id: 't2', nodeId: 'formation' });

        expect(panel.tabs.length).toBe(2);
        expect(panel.tabs[0]).toMatchObject({ id: 'dag', nodeId: null, label: 'DAG', kind: 'dag' });
        expect(result).toBe(panel.tabs[1]);
        expect(panel.activeTabId).toBe('t2');
    });

    it('openOrActivateTab adds a new tab instead of mutating a non-pinned dag-kind active tab', () => {
        const panel = new Panel();
        panel.addTab(new Tab({ id: 'dag', nodeId: null, label: 'DAG', kind: 'dag' }));

        panel.openOrActivateTab({ id: 't2', nodeId: 'formation' });

        expect(panel.tabs.length).toBe(2);
        expect(panel.tabs[0].kind).toBe('dag');
        expect(panel.tabs[1].nodeId).toBe('formation');
    });

    it('openOrActivateTab switches to an existing tab for the same nodeId instead of creating a duplicate', () => {
        const panel = new Panel();
        panel.addTab(new Tab({ id: 't1', nodeId: 'formation' })); // active
        panel.addTab(new Tab({ id: 't2', nodeId: 'genus' }));     // active (addTab activates last-added)
        panel.activateTab('t1'); // formation active again, genus open but not active

        const result = panel.openOrActivateTab({ id: 't3', nodeId: 'genus' });

        expect(panel.tabs.length).toBe(2); // no duplicate, no third tab
        expect(panel.tabs.map((t) => t.nodeId)).toEqual(['formation', 'genus']);
        expect(panel.activeTabId).toBe('t2'); // switched to the existing genus tab
        expect(result).toBe(panel.tabs[1]);
    });

    // canCloseTab drives the tab strip's × visibility — only pinned tabs are exempt

    it('canCloseTab is false for a pinned tab even with other tabs present', () => {
        const panel = new Panel();
        panel.addTab(new Tab({ id: 't1', nodeId: null, label: 'DAG', kind: 'dag', pinned: true }));
        panel.addTab(new Tab({ id: 't2', nodeId: 'formation' }));
        expect(panel.canCloseTab('t1')).toBe(false);
    });

    it('canCloseTab is true for a non-pinned tab when another tab exists', () => {
        const panel = new Panel();
        panel.addTab(new Tab({ id: 't1', nodeId: null, label: 'DAG', kind: 'dag', pinned: true }));
        panel.addTab(new Tab({ id: 't2', nodeId: 'formation' }));
        expect(panel.canCloseTab('t2')).toBe(true);
    });

    it('canCloseTab is true for a non-pinned tab even when it is the only tab in the panel', () => {
        const panel = new Panel();
        panel.addTab(new Tab({ id: 't1', nodeId: 'formation' }));
        expect(panel.canCloseTab('t1')).toBe(true);
    });

    it('closeTab refuses to remove a pinned tab', () => {
        const panel = new Panel();
        panel.addTab(new Tab({ id: 't1', nodeId: null, label: 'DAG', kind: 'dag', pinned: true }));
        panel.addTab(new Tab({ id: 't2', nodeId: 'formation' }));
        panel.closeTab('t1');
        expect(panel.tabs.map((t) => t.id)).toEqual(['t1', 't2']);
    });
});

describe('SplitContainer', () => {
    it('stores a direction and exactly two children', () => {
        const a = new Panel();
        const b = new Panel();
        const split = new SplitContainer({ direction: 'row', children: [a, b] });
        expect(split.direction).toBe('row');
        expect(split.children).toEqual([a, b]);
    });

    it('accepts another SplitContainer as a child (nested splits)', () => {
        const a = new Panel();
        const b = new Panel();
        const c = new Panel();
        const inner = new SplitContainer({ direction: 'row', children: [a, b] });
        const outer = new SplitContainer({ direction: 'column', children: [inner, c] });
        expect(outer.children[0]).toBe(inner);
        expect(outer.children[1]).toBe(c);
    });
});

describe('Window', () => {
    it('starts with a single empty Panel as its root', () => {
        const win = new Window();
        expect(win.root).toBeInstanceOf(Panel);
        expect(win.root.tabs).toEqual([]);
    });

    it('getActivePanel returns the root panel', () => {
        const win = new Window();
        expect(win.getActivePanel()).toBe(win.root);
    });

    it('getActivePanel reflects tabs added to the root panel', () => {
        const win = new Window();
        win.getActivePanel().openOrActivateTab({ id: 't1', nodeId: 'formation' });
        expect(win.root.tabs.length).toBe(1);
        expect(win.getActivePanel().tabs[0].nodeId).toBe('formation');
    });

    it('getAllPanels returns just the root when nothing has split', () => {
        const win = new Window();
        expect(win.getAllPanels()).toEqual([win.root]);
    });

    it('splitActivePanel replaces the root Panel with a SplitContainer holding the original and a new panel', () => {
        const win = new Window();
        const original = win.root;
        const newPanel = win.splitActivePanel('row');
        expect(win.root).toBeInstanceOf(SplitContainer);
        expect(win.root.direction).toBe('row');
        expect(win.root.children).toEqual([original, newPanel]);
        expect(newPanel).toBeInstanceOf(Panel);
        expect(newPanel).not.toBe(original);
    });

    it('splitActivePanel makes the new panel the active one', () => {
        const win = new Window();
        const newPanel = win.splitActivePanel('row');
        expect(win.getActivePanel()).toBe(newPanel);
    });

    it('getAllPanels returns both panels after a split', () => {
        const win = new Window();
        const original = win.root;
        const newPanel = win.splitActivePanel('row');
        expect(win.getAllPanels()).toEqual([original, newPanel]);
    });

    it('splitActivePanel again splits whichever panel is currently active, nesting the tree', () => {
        const win = new Window();
        const first = win.root;
        const second = win.splitActivePanel('row');
        const third = win.splitActivePanel('column');
        expect(win.getAllPanels()).toEqual([first, second, third]);
        expect(win.getActivePanel()).toBe(third);
        // second was replaced in-place by a new SplitContainer([second, third])
        expect(win.root.children[1]).toBeInstanceOf(SplitContainer);
        expect(win.root.children[1].children).toEqual([second, third]);
    });

    it('splitActivePanel has no upper limit on panel count (matches VS Code)', () => {
        const win = new Window();
        for (let i = 0; i < 8; i++) {
            const result = win.splitActivePanel('row');
            expect(result).not.toBe(null);
        }
        expect(win.getAllPanels().length).toBe(9);
    });

    it('focusPanel switches the active panel to an existing panel id', () => {
        const win = new Window();
        const original = win.root;
        win.splitActivePanel('row'); // active is now the new panel
        win.focusPanel(original.id);
        expect(win.getActivePanel()).toBe(original);
    });

    it('focusPanel is a no-op for an unknown panel id', () => {
        const win = new Window();
        const newPanel = win.splitActivePanel('row');
        win.focusPanel('does-not-exist');
        expect(win.getActivePanel()).toBe(newPanel);
    });

    it('closeTab removes an empty panel and promotes its sibling, collapsing back to a single root Panel', () => {
        const win = new Window();
        const original = win.root;
        original.openOrActivateTab({ id: 't1', nodeId: 'formation' });
        const newPanel = win.splitActivePanel('row');
        newPanel.openOrActivateTab({ id: 't2', nodeId: 'genus' });

        win.closeTab(newPanel.id, 't2');

        expect(win.root).toBe(original);
        expect(win.getAllPanels()).toEqual([original]);
    });

    it('closeTab on the only panel leaves it in place, even once empty', () => {
        const win = new Window();
        const original = win.root;
        original.openOrActivateTab({ id: 't1', nodeId: 'formation' });

        win.closeTab(original.id, 't1');

        expect(win.root).toBe(original);
        expect(win.root.tabs).toEqual([]);
    });

    it('closeTab on a panel with remaining tabs does not remove the panel', () => {
        const win = new Window();
        const original = win.root;
        original.addTab(new Tab({ id: 't1', nodeId: 'formation' }));
        original.addTab(new Tab({ id: 't2', nodeId: 'genus' }));
        win.splitActivePanel('row');

        win.closeTab(original.id, 't1');

        expect(win.getAllPanels().length).toBe(2);
        expect(original.tabs.length).toBe(1);
    });

    it('after promoting a sibling, the promoted panel becomes active if the closed panel was active', () => {
        const win = new Window();
        const original = win.root;
        const newPanel = win.splitActivePanel('row'); // newPanel is active
        newPanel.openOrActivateTab({ id: 't1', nodeId: 'formation' });

        win.closeTab(newPanel.id, 't1');

        expect(win.getActivePanel()).toBe(original);
    });
});