// Hover text and screen-reader label for a collapse/expand button: what
// pressing it will do, named for what it collapses ("panel", "sidebar", "DAG").
function collapseToggleLabels(collapsed, subject) {
    const text = `${collapsed ? 'Expand' : 'Collapse'} ${subject}`;
    return {title: text, label: text};
}

function setCollapseToggleLabels(button, collapsed, subject) {
    if (!button) return;
    const {title, label} = collapseToggleLabels(collapsed, subject);
    button.title = title;
    button.setAttribute('aria-label', label);
}

// A panel beside another one collapses to a thin vertical bar; one stacked
// above or below another collapses to a thin horizontal bar.
function panelCollapseOrientation(parent) {
    return parent && parent.classList.contains('split-row') ? 'vertical' : 'horizontal';
}

// The twisty from VS Code's sections: down while the panel is open, right once it is
// collapsed. The same whichever way the panels are laid out.
function collapseChevron(collapsed) {
    return collapsed ? 'fa-chevron-right' : 'fa-chevron-down';
}

// What a panel's collapse button shows and whether it works. The only open panel has
// nowhere to give its space to, so it cannot be collapsed (VS Code always keeps one
// thing open); a collapsed panel can always be opened again.
function collapseButtonState(collapsed, isOnlyOpenPanel) {
    if (!collapsed && isOnlyOpenPanel) return {disabled: true, title: "Can't collapse the only open panel"};
    return {disabled: false, title: collapsed ? 'Expand panel' : 'Collapse panel'};
}

// Hover text of the sidebar toggle in the toolbar: VS Code's wording plus the
// shortcut for the platform.
function sidebarToggleTitle(isMac) {
    return `Toggle Primary Side Bar (${isMac ? '\u2318B' : 'Ctrl+B'})`;
}

// Cmd+B (Mac) or Ctrl+B, as in VS Code — not with Shift or Alt added.
function isSidebarShortcut(e) {
    return (e.metaKey || e.ctrlKey) && !e.shiftKey && !e.altKey &&
        typeof e.key === 'string' && e.key.toLowerCase() === 'b';
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        collapseToggleLabels, setCollapseToggleLabels, sidebarToggleTitle, isSidebarShortcut,
        panelCollapseOrientation, collapseChevron, collapseButtonState,
    };
}
