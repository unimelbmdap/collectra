// Tab / Panel — DOM-free state for the split-panel/tab system (see
// docs/superpowers/specs/2026-08-05-split-panels-design.md). Window,
// SplitContainer, and NodeContentModel are deliberately not built yet —
// they're only needed once panels can actually be split ("Open to the
// Side"), which isn't implemented yet either. Loaded as a plain classic
// script in index.html (defines globals, same as everything else there),
// and via require() in tests/.

class Tab {
    constructor({ id, nodeId, label, kind = 'node', pinned = false }) {
        this.id = id;
        this.nodeId = nodeId;
        this.label = label ?? nodeId;
        this.kind = kind;
        this.pinned = pinned;
    }
}

let nextPanelId = 1;

class Panel {
    constructor() {
        this.id = 'panel' + nextPanelId++;
        this.tabs = [];
        this.activeTabId = null;
    }

    addTab(tab) {
        this.tabs.push(tab);
        this.activeTabId = tab.id;
        return tab;
    }

    hasTab(nodeId) {
        return this.tabs.some((t) => t.nodeId === nodeId);
    }

    getTab(tabId) {
        return this.tabs.find((t) => t.id === tabId) || null;
    }

    getActiveTab() {
        return this.activeTabId === null ? null : this.getTab(this.activeTabId);
    }

    activateTab(tabId) {
        if (this.getTab(tabId)) {
            this.activeTabId = tabId;
        }
        return this.activeTabId;
    }

    // Only a pinned tab is ever non-closeable — closing the last tab in a
    // panel is allowed and just leaves it empty.
    canCloseTab(tabId) {
        const tab = this.getTab(tabId);
        if (!tab || tab.pinned) return false;
        return true;
    }

    closeTab(tabId) {
        this.removeTab(tabId);
    }

    // Same as closeTab, but returns the Tab instead of discarding it.
    removeTab(tabId) {
        const tab = this.getTab(tabId);
        if (!tab || tab.pinned) return undefined;
        const index = this.tabs.findIndex((t) => t.id === tabId);
        if (index === -1) return undefined;
        this.tabs.splice(index, 1);
        if (this.activeTabId === tabId) {
            const fallback = this.tabs[index] || this.tabs[index - 1] || null;
            this.activeTabId = fallback ? fallback.id : null;
        }
        return tab;
    }

    // Unlike addTab, inserts at a specific position instead of appending.
    insertTab(tab, index) {
        this.tabs.splice(index, 0, tab);
        this.activeTabId = tab.id;
    }

    // Closes every tab whose (nodeId, kind) fails isValid(nodeId, kind) —
    // used on page switch to drop tabs for nodes that don't exist on the
    // new page, instead of leaving them open showing stale content.
    // Non-node tabs (kind !== 'node', e.g. a DAG tab) pass their kind
    // through so a caller's predicate can exempt them instead of failing
    // every check against a null nodeId.
    pruneTabs(isValid) {
        this.tabs = this.tabs.filter((t) => isValid(t.nodeId, t.kind));
        if (this.activeTabId && !this.tabs.some((t) => t.id === this.activeTabId)) {
            this.activeTabId = this.tabs[0] ? this.tabs[0].id : null;
        }
    }

    // Plain-click behavior: reuse the currently active tab, replacing its
    // content, rather than creating a new one — matches today's
    // single-panel behavior and VS Code's default (new tabs only come from
    // "Open in New Tab", not from clicking a node). If nodeId is already
    // open in a *different* tab in this panel, switches to that tab
    // instead of mutating the active one — a node is only ever open once
    // per panel at a time, never duplicated across tabs.
    //
    // Exception: a pinned or non-node active tab (e.g. DAG) is never
    // mutated — adds a new tab instead of repurposing it.
    openOrActivateTab({ id, nodeId, label }) {
        const existing = this.tabs.find((t) => t.nodeId === nodeId);
        if (existing) {
            this.activateTab(existing.id);
            return existing;
        }
        const active = this.getActiveTab();
        if (!active || active.pinned || active.kind !== 'node') {
            return this.addTab(new Tab({ id, nodeId, label }));
        }
        active.nodeId = nodeId;
        active.label = label ?? nodeId;
        return active;
    }
}

// Arranges exactly two children — each either a Panel or another
// SplitContainer — side by side ('row') or stacked ('column'). Never holds
// tabs itself; it's purely the "how are these two things arranged"
// wrapper. Nesting (a child that's itself a SplitContainer) is what lets
// splitting apply to any existing panel repeatedly, not just the root.
class SplitContainer {
    constructor({ direction, children }) {
        this.direction = direction;
        this.children = children;
    }
}

// Owns the content area's layout root — either a single Panel (nothing
// split yet) or a SplitContainer (once something has). getActivePanel()
// is the seam every caller (selectNode, tab-strip rendering, etc.) should
// go through instead of touching root directly, so callers don't need to
// change as the tree grows deeper.
//
// Naming note: this shadows the browser's built-in `Window` interface for
// any *unqualified* `Window` reference after this declaration (class
// declarations don't attach to `window`/`globalThis`, so `window.Window`
// still resolves to the native one). Nothing in this codebase does
// `instanceof Window` or otherwise relies on the bare global, so this is
// safe here, but worth knowing if that ever changes.
class Window {
    constructor() {
        this.root = new Panel();
        this.activePanelId = this.root.id;
    }

    getAllPanels(node = this.root) {
        if (node instanceof SplitContainer) {
            return node.children.flatMap((child) => this.getAllPanels(child));
        }
        return [node];
    }

    getActivePanel() {
        return this.getAllPanels().find((p) => p.id === this.activePanelId) || null;
    }

    focusPanel(panelId) {
        if (this.getAllPanels().some((p) => p.id === panelId)) {
            this.activePanelId = panelId;
        }
        return this.activePanelId;
    }

    // Splits whichever panel is currently active into two: the original
    // panel plus a new empty one, arranged in `direction`. The new panel
    // becomes active. No upper limit on panel count, same as VS Code —
    // screen space is the only practical constraint.
    splitActivePanel(direction) {
        const activePanel = this.getActivePanel();
        const newPanel = new Panel();
        const container = new SplitContainer({ direction, children: [activePanel, newPanel] });
        this._replaceNode(activePanel, container);
        this.activePanelId = newPanel.id;
        return newPanel;
    }

    // Closes a tab in the given panel. If that empties the panel and it
    // isn't the last panel left, the panel is removed and its sibling
    // takes its place in the tree — collapsing a SplitContainer back down
    // once only one side of it has content. The very last panel is never
    // removed, even once empty (there's always somewhere to render into).
    closeTab(panelId, tabId) {
        const panel = this.getAllPanels().find((p) => p.id === panelId);
        if (!panel) return;
        panel.closeTab(tabId);
        if (panel.tabs.length > 0) return;
        if (panel === this.root) return;
        const parent = this._findParent(panel);
        if (!parent) return;
        const sibling = parent.children.find((c) => c !== panel);
        this._replaceNode(parent, sibling);
        if (this.activePanelId === panelId) {
            const fallback = this.getAllPanels()[0];
            this.activePanelId = fallback ? fallback.id : null;
        }
    }

    // Replaces `target` with `replacement` wherever it sits in the tree
    // (root itself, or as one child of some SplitContainer), mutating in
    // place.
    _replaceNode(target, replacement) {
        if (this.root === target) {
            this.root = replacement;
            return;
        }
        const parent = this._findParent(target);
        if (!parent) return;
        parent.children[parent.children.indexOf(target)] = replacement;
    }

    _findParent(target, node = this.root) {
        if (!(node instanceof SplitContainer)) return null;
        if (node.children.includes(target)) return node;
        for (const child of node.children) {
            const found = this._findParent(target, child);
            if (found) return found;
        }
        return null;
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { Tab, Panel, SplitContainer, Window };
}