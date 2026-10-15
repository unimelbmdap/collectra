// Remembering which tabs are open in which panel. Pure state in, state out:
// the page decides when to save and when to restore.

function tabClass() {
    return typeof Tab !== 'undefined' ? Tab : require('./panelSystem.js').Tab;
}

// panels: {key: Panel}. labelOf(nodeId) -> DAG label, for a tab that was
// opened without one. The result is what the backend stores as YAML.
function serializeArrangement(panels, focused, labelOf, {layout = [], sizes = {}, collapsed} = {}) {
    const result = {version: 1, focused, layout, sizes, panels: {}};
    if (collapsed) result.collapsed = collapsed; // omitted when not given, so an old save stays "unknown"
    for (const [key, panel] of Object.entries(panels)) {
        const tabs = [];
        let active = 0;
        for (const tab of panel.tabs) {
            let saved = null;
            if (tab.kind === 'dag') {
                saved = {kind: 'dag'};
            } else if (tab.kind === 'node') {
                const group = tab.groupKey || (tab.nodeId ? labelOf(tab.nodeId) : null);
                if (group) saved = {kind: 'node', group};
            }
            if (!saved) continue;
            if (tab.id === panel.activeTabId) active = tabs.length;
            tabs.push(saved);
        }
        result.panels[key] = {active, tabs};
    }
    return result;
}

// Redoes the user's panel splits and closes, in order, through the same
// actions they clicked. page.split(from, direction) creates a panel and
// returns the key it really got; page.remove(key) closes one; page.has(key)
// says whether a panel exists. Returns {savedKey: actualKey} so the saved
// tabs of each panel can be put back on the right one. top and bottom always
// exist and map to themselves.
function replayLayout(ops, page) {
    const map = {top: 'top', bottom: 'bottom'};
    for (const op of Array.isArray(ops) ? ops : []) {
        if (op.op === 'split') {
            const from = map[op.from];
            if (!from || !page.has(from)) continue;
            map[op.key] = page.split(from, op.direction);
        } else if (op.op === 'remove') {
            const key = map[op.key];
            if (!key || !page.has(key)) continue;
            page.remove(key);
            delete map[op.key];
        }
    }
    return map;
}

// A panel must always stay open: when every panel is collapsed (for instance after a restore
// that removed the one that was open) this says which to open — the first that is not the
// top DAG panel, else the first. null when something is already open or there are no panels.
function keepOneExpanded(keys, isCollapsed) {
    if (keys.length === 0 || keys.some((key) => !isCollapsed(key))) return null;
    const preferred = keys.find((key) => key !== 'top');
    return preferred !== undefined ? preferred : keys[0];
}

const SAVED_SIZE = /^0 [01] ([\d.]+)(px|%)$/;

// A saved panel size as a shrinkable share of its container ("0 1 50.00%"), so
// panels keep their proportions when the window changes size, as in VS Code.
// Old pixel sizes are converted against the container's size now. Never more
// than 90% or less than 10%, so no panel can be squeezed out.
function sizeToPercent(flex, containerPx) {
    const match = SAVED_SIZE.exec(flex || '');
    if (!match) return null;
    let percent = match[2] === '%'
        ? parseFloat(match[1])
        : (containerPx > 0 ? parseFloat(match[1]) / containerPx * 100 : null);
    if (percent === null || !Number.isFinite(percent)) return null;
    percent = Math.min(90, Math.max(10, percent));
    return `0 1 ${percent.toFixed(2)}%`;
}

// Replaces the panel's tabs with the saved ones. A node tab shows the first
// instance this page has of its node, or stays blank (nodeId null) when the
// page has none. A missing spec leaves the panel alone; a saved panel with no
// tabs is emptied, as the user left it.
function restorePanelTabs(panel, spec, {makeId, instancesFor, labelFor}) {
    if (!spec || !Array.isArray(spec.tabs)) return;
    const TabType = tabClass();
    const seen = new Set();
    const tabs = [];
    let activeId = null;
    spec.tabs.forEach((saved, index) => {
        let tab;
        if (saved.kind === 'dag') {
            if (seen.has('dag')) return;
            seen.add('dag');
            tab = new TabType({id: makeId(), nodeId: null, label: 'DAG', kind: 'dag'});
        } else if (saved.kind === 'node' && saved.group) {
            if (seen.has(saved.group)) return;
            seen.add(saved.group);
            const instances = instancesFor(saved.group) || [];
            tab = new TabType({
                id: makeId(),
                nodeId: instances.length ? instances[0] : null,
                label: labelFor(saved.group),
                groupKey: saved.group,
            });
        } else {
            return;
        }
        if (index === spec.active) activeId = tab.id;
        tabs.push(tab);
    });
    panel.tabs = tabs;
    panel.activeTabId = activeId || (tabs[0] ? tabs[0].id : null);
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {serializeArrangement, restorePanelTabs, replayLayout, sizeToPercent, keepOneExpanded};
}
