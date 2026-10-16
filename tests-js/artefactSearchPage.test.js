const html = require('./displaySources.js');
const between = (from, to) => html.slice(html.indexOf(from), html.indexOf(to, html.indexOf(from)));

describe('opening the search', () => {
    it('has a magnifier button before the DAG icon that opens the bar', () => {
        const toggle = html.indexOf('id="artefactSearchToggle"');
        expect(toggle).toBeGreaterThan(-1);
        expect(toggle).toBeLessThan(html.indexOf('id="dagToolbarBtn"'));
        const tag = html.slice(html.lastIndexOf('<button', toggle), html.indexOf('</button>', toggle));
        expect(tag).toContain('onclick="toggleArtefactSearch()"');
        expect(tag).toContain('fa-search');
        expect(tag).toContain('aria-label="Search artefacts"');
    });

    it('opens with Cmd/Ctrl+F, but not over an open dialog', () => {
        const at = html.indexOf('isSearchShortcut(e)');
        expect(at).toBeGreaterThan(-1);
        const listener = html.slice(html.lastIndexOf("document.addEventListener('keydown'", at), html.indexOf('});', at));
        expect(listener).toContain("document.querySelector('.modal-overlay.visible')");
        expect(listener).toContain('openArtefactSearch();');
        expect(html).not.toContain("e.key.toLowerCase() === 'f'");
    });

    it('loads the search logic', () => {
        expect(html).toContain('<script src="artefactSearch.js"></script>');
    });
});

describe('the bar', () => {
    const css = html.slice(html.indexOf('.artefact-search {'), html.indexOf('.artefact-search-field {'));
    const bar = html.slice(html.indexOf('<div class="artefact-search"'), html.indexOf('id="dagToolbarBtn"'));
    const at = (id) => bar.indexOf(id);

    it('is hidden until opened, so no Done or arrows show on an empty window', () => {
        expect(css).toContain('display: none;');
        expect(html).toContain('.artefact-search.open {');
        expect(bar.slice(0, bar.indexOf('>'))).not.toContain('open');
    });

    it('is centred at the top of the window', () => {
        expect(css).toContain('position: fixed;');
        expect(css).toContain('left: 50%;');
        expect(css).toContain('transform: translateX(-50%);');
    });

    it('has a magnifier, the box, the count and a clear button in one field, in that order', () => {
        expect(bar).toContain('class="artefact-search-field"');
        expect(at('fa-search')).toBeLessThan(at('id="artefactSearchInput"'));
        expect(at('id="artefactSearchInput"')).toBeLessThan(at('id="artefactSearchCount"'));
        expect(at('id="artefactSearchCount"')).toBeLessThan(at('id="artefactSearchClear"'));
    });

    it('has the arrows as one joined control after the field, then Done', () => {
        expect(at('id="artefactSearchClear"')).toBeLessThan(at('class="artefact-search-steps"'));
        expect(at('id="artefactSearchPrev"')).toBeLessThan(at('id="artefactSearchNext"'));
        expect(at('id="artefactSearchNext"')).toBeLessThan(at('id="artefactSearchDone"'));
    });

    it('draws the arrows as icons, not text characters', () => {
        const steps = bar.slice(at('class="artefact-search-steps"'), at('id="artefactSearchDone"'));
        expect(steps).toContain('fa-chevron-left');
        expect(steps).toContain('fa-chevron-right');
        expect(steps).not.toContain('&#8249;');
    });

    it('never paints a filled grey block when the arrows are disabled', () => {
        const disabled = html.slice(html.indexOf('.artefact-search-steps button:disabled'), html.indexOf('}', html.indexOf('.artefact-search-steps button:disabled')));
        expect(disabled).toContain('background: transparent;');
    });

    it('has no match-mode dropdown', () => {
        expect(bar).not.toContain('<select');
    });

    it('centres the magnifier on the box', () => {
        const icon = html.slice(html.indexOf('.artefact-search-icon {'), html.indexOf('}', html.indexOf('.artefact-search-icon {')));
        expect(icon).toContain('top: 50%;');
        expect(icon).toContain('translateY(-50%)');
    });
});

describe('search state', () => {
    // Tab strips are drawn while the script loads and read this state.
    it('is declared before the first tab strip is drawn', () => {
        const firstDraw = html.indexOf('panelControllers.bottom.renderTabStripView();');
        for (const name of ['searchLabels', 'searchState']) {
            const declared = html.indexOf(`let ${name}`);
            expect(declared, name).toBeGreaterThan(-1);
            expect(declared, name).toBeLessThan(firstDraw);
        }
    });
});

describe('highlights', () => {
    it('marks tabs of matching nodes in every panel', () => {
        expect(between('function renderTabStrip(', 'function activateTabInPanel')).toContain("searchLabels.has(tab.groupKey)");
        expect(html).toContain('.tab.search-hit');
    });

    it('leaves the DAG alone', () => {
        expect(between('syncDagHighlights() {', 'PanelController.prototype.clearPageArtefact')).not.toContain('search-hit');
        expect(html).not.toContain("selector: 'node.search-hit'");
    });

    it('marks the matched words inside a shown text artefact', () => {
        expect(between('PanelController.prototype.renderTextNodeAsPage', 'PanelController.prototype.clear = ')).toContain('markSearchInEditor(this);');
        const mark = between('function markSearchInEditor(', 'async function stepArtefactSearch');
        expect(mark).toContain('findRanges(');
        expect(mark).toContain('markText(');
        expect(html).toContain('.search-mark');
    });
});

describe('behaviour', () => {
    it('runs after a short pause in typing and ignores answers to an older query', () => {
        const run = between('function handleArtefactSearchInput()', 'async function stepArtefactSearch');
        expect(run).toContain('setTimeout(');
        expect(run).toContain('search_artefacts(');
        expect(run).toContain('searchState.token');
    });

    it('stepping opens the match in the panel you chose, or the topmost visible one by default', () => {
        const step = between('async function stepArtefactSearch(', '// ── END ARTEFACT SEARCH');
        expect(step).toContain('stepIndex(');
        expect(step).toContain('searchTargetPanel(Object.keys(panelControllers), panelFocusChosen ? focusedPanel : null, panelIsVisible, panelPosition)');
        expect(step).toContain('panelControllers[key].openOrActivateTab(');
        expect(step).not.toContain('panelControllers[focusedPanel]');
    });

    it('Enter steps forward, Shift+Enter back, Escape clears', () => {
        const keys = between('function handleArtefactSearchKey(', 'function handleArtefactSearchInput()');
        expect(keys).toContain("e.key === 'Enter'");
        expect(keys).toContain('e.shiftKey ? -1 : 1');
        expect(keys).toContain("e.key === 'Escape'");
    });

    it('searches again after a page opens and after a delete', () => {
        expect(between('async function loadCollectraFolderInner(index)', '// Collapse the sidebar')).toContain('rerunArtefactSearch();');
        expect(between('async function deleteNodeWithConfirm(', 'function showResult(')).toContain('rerunArtefactSearch();');
    });
});

describe('the search box is not restyled by the pages search box', () => {
    // .folder-search-input is declared later in the file with its own margin
    // and padding, so a single-class rule here would lose to it.
    const rule = (selector) => {
        const start = html.indexOf(`        ${selector} {`);
        expect(start, selector).toBeGreaterThan(-1);
        return html.slice(start, html.indexOf('}', start));
    };

    it('out-ranks .folder-search-input for margin and padding', () => {
        const input = rule('.artefact-search .artefact-search-input');
        expect(input).toContain('margin: 0;');
        expect(input).toContain('padding: 0 32px 0 28px;');
        expect(input).toContain('height: 26px;');
    });

    it('has its own focus ring instead of the browser default', () => {
        const focus = rule('.artefact-search .artefact-search-input:focus');
        expect(focus).toContain('outline: none;');
        expect(focus).toContain('border-color: var(--app-primary');
    });
});

it('hides the browser\'s own clear button so only ours shows', () => {
    expect(html).toContain('.artefact-search .artefact-search-input::-webkit-search-cancel-button');
    const start = html.indexOf('.artefact-search .artefact-search-input::-webkit-search-cancel-button');
    expect(html.slice(start, html.indexOf('}', start))).toContain('display: none;');
});

describe('the box is a plain text box', () => {
    const input = () => {
        const at = html.indexOf('id="artefactSearchInput"');
        return html.slice(html.lastIndexOf('<input', at), html.indexOf('>', at));
    };

    it('is not type="search", so the browser adds no clear button or styling of its own', () => {
        expect(input()).toContain('type="text"');
        expect(input()).not.toContain('type="search"');
        expect(input()).toContain('role="searchbox"');
    });
});

describe('the search marker on a tab', () => {
    const rule = (selector) => {
        const start = html.indexOf(`        ${selector} {`);
        expect(start, selector).toBeGreaterThan(-1);
        return html.slice(start, html.indexOf('}', start));
    };

    it('has its own rule, strong enough to apply on any tab', () => {
        expect(rule('.tab-strip .tab.search-hit')).toContain('box-shadow: inset 0 -3px 0 #facc15;');
        expect(html).not.toContain('        .tab.search-hit {');
    });

    it('is not lost on the active tab of the focused panel, which has its own accent', () => {
        const both = rule('.panel-focused .tab-strip .tab.active.search-hit');
        expect(both).toContain('inset 0 2px 0 rgba(var(--app-primary-rgb');
        expect(both).toContain('inset 0 -3px 0 #facc15');
    });
});
