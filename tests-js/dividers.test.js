const fs = require('node:fs');
const html = fs.readFileSync(require.resolve('../collectra/gui/index.html'), 'utf8');
const rule = (selector) => {
    const start = html.indexOf(`        ${selector} {`);
    expect(start, selector).toBeGreaterThan(-1);
    return html.slice(start, html.indexOf('}', start));
};

describe('dividers between the sidebar, toolbar and panels', () => {
    it('draws no second ring around the focused panel (it doubled up the toolbar line)', () => {
        expect(html).not.toContain('        .panel-focused {');
        expect(html).not.toMatch(/\.panel-focused\s*\{[^}]*inset 0 0 0 1px/);
    });

    it('uses one hairline colour for every seam', () => {
        expect(rule('.sidebar-top')).toContain('border-bottom: 1px solid var(--app-divider');
        expect(rule('.global-toolbar-row')).toContain('border-bottom: 1px solid var(--app-divider');
        expect(rule('.sidebar-resize-handle::after')).toContain('background: var(--app-divider');
    });

});

describe('tab strips', () => {
    it('every panel has a hairline under its tabs, in the same colour as the other dividers', () => {
        expect(rule('.tab-strip-row')).toContain('border-bottom: 1px solid var(--app-divider');
    });

    it('the active tab sits on that line, joined to the panel below it', () => {
        const active = rule('.tab-strip .tab.active');
        expect(active).toContain('position: relative;');
        expect(active).toContain('margin-bottom: -1px;');
        expect(active).toMatch(/border-bottom:\s*1px solid var\(--app-surface/);
    });

    it('no heavy underline marks the focused panel any more', () => {
        expect(html).not.toContain('.panel-focused .tab-strip-row {');
        expect(html).not.toMatch(/\.panel-focused[^{]*\{[^}]*inset 0 -1px/);
    });

    it('the focused panel is shown by a thin accent on top of its active tab', () => {
        const focus = rule('.panel-focused .tab-strip .tab.active');
        expect(focus).toContain('box-shadow: inset 0 2px 0');
        expect(focus).toContain('--app-primary-rgb');
    });
});

describe('no styles for things that no longer exist', () => {
    it.each(['.panel-collapse-toggle', '.pager-delete', '.tab-node-id'])('%s', (name) => {
        expect(html).not.toContain(name);
    });

    it('the search banner says what it really searches', () => {
        expect(html).toContain('ARTEFACT SEARCH (toolbar box: finds text and names)');
        expect(html).not.toContain('finds text, names, labels, ids');
    });
});
