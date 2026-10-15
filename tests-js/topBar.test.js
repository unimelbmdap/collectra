const fs = require('node:fs');
const html = fs.readFileSync(require.resolve('../collectra/gui/index.html'), 'utf8');
const between = (from, to) => html.slice(html.indexOf(from), html.indexOf(to, html.indexOf(from)));
const rule = (selector) => {
    const start = html.indexOf(`        ${selector} {`);
    expect(start, selector).toBeGreaterThan(-1);
    return html.slice(start, html.indexOf('}', start));
};

describe('nothing empty above the content', () => {
    it('has no reserved strip at the top of the page', () => {
        expect(html).not.toContain('titlebar-strip');
        expect(html).not.toContain('pywebview-drag-region');
    });
});

describe('the sidebar toggle in the toolbar', () => {
    const toolbar = html.slice(html.indexOf('<div class="global-toolbar-row">'), html.indexOf('id="dagToolbarBtn"'));
    const button = toolbar.slice(toolbar.lastIndexOf('<button', toolbar.indexOf('id="sidebarToggleBtn"')), toolbar.indexOf('</button>', toolbar.indexOf('id="sidebarToggleBtn"')));

    it('is the first thing in the toolbar, at its left edge', () => {
        expect(toolbar.indexOf('id="sidebarToggleBtn"')).toBeGreaterThan(-1);
        expect(toolbar.indexOf('id="sidebarToggleBtn"')).toBeLessThan(toolbar.indexOf('id="artefactSearchBar"'));
        expect(rule('#sidebarToggleBtn')).toContain('margin-right: auto;');
    });

    it('toggles the sidebar and is named like the VS Code one', () => {
        expect(button).toContain('onclick="toggleSidebarCollapse()"');
        expect(button).toContain('title="Toggle Primary Side Bar"');
        expect(button).toContain('aria-label="Toggle Primary Side Bar"');
    });

    it('draws a window with a sidebar on the left', () => {
        expect(button).toContain('<rect');
        expect(button).toContain('stroke="currentColor"');
    });

    it('adds the shortcut for the platform to its hover text when the page starts', () => {
        expect(html).toContain('sidebarToggleTitle(');
        expect(html).toContain("/Mac|iPhone|iPad/.test(navigator.platform)");
    });

    it('replaces the little chevron on the sidebar', () => {
        expect(html).not.toContain('sidebarCollapseToggle');
        expect(html).not.toContain('sidebar-collapse-toggle');
    });
});

describe('collapsing the sidebar', () => {
    it('hides it completely, as VS Code does', () => {
        expect(rule('.app-container.sidebar-collapsed')).toContain('grid-template-columns: 0 1fr;');
    });

    it('shows the toolbar button as pressed while the sidebar is open', () => {
        const toggle = between('function toggleSidebarCollapse()', '\n        }\n');
        expect(toggle).toContain("document.getElementById('sidebarToggleBtn').classList.toggle('active', !collapsed);");
        expect(toggle).not.toContain('setCollapseToggleLabels');
    });

    it('starts shown as pressed', () => {
        expect(html).toMatch(/id="sidebarToggleBtn"[^>]*class="[^"]*active|class="[^"]*active[^"]*"[^>]*id="sidebarToggleBtn"/);
    });
});

describe('Cmd/Ctrl+B', () => {
    it('toggles the sidebar from anywhere, ahead of the text editor\'s own bold shortcut', () => {
        const at = html.indexOf('isSidebarShortcut(e)');
        expect(at).toBeGreaterThan(-1);
        const listener = html.slice(html.lastIndexOf("document.addEventListener('keydown'", at), html.indexOf('});', at) + 3);
        expect(listener).toContain('e.preventDefault();');
        expect(listener).toContain('toggleSidebarCollapse();');
        expect(listener).toContain('true);'); // capture phase, so the editor never sees it
    });
});

describe('the sidebar starts higher too', () => {
    it('has less room above the logo and buttons', () => {
        expect(rule('.sidebar-top')).toContain('padding: 10px 15px 15px;');
    });
});
