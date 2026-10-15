// Runs the real search functions from the page against stand-in elements, so
// what the bar does (not just what its markup says) is tested.
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync(require.resolve('../collectra/gui/index.html'), 'utf8');
const search = require('../collectra/gui/artefactSearch.js');
const slice = (from, to) => html.slice(html.indexOf(from), html.indexOf(to, html.indexOf(from)));
const start = html.indexOf('// ── ARTEFACT SEARCH');
const source = html.slice(start, html.indexOf('// ── END ARTEFACT SEARCH', start));

function element(extra = {}) {
    const classes = new Set();
    return {
        value: '', textContent: '', hidden: false, disabled: false, focused: false,
        classList: {
            add: (c) => classes.add(c), remove: (c) => classes.delete(c),
            toggle: (c, on) => { (on === undefined ? !classes.has(c) : on) ? classes.add(c) : classes.delete(c); },
            contains: (c) => classes.has(c),
        },
        focus() { this.focused = true; }, blur() { this.focused = false; }, select() {},
        ...extra,
    };
}

function page(matches = []) {
    const ids = ['artefactSearchBar', 'artefactSearchInput', 'artefactSearchCount', 'artefactSearchClear',
        'artefactSearchPrev', 'artefactSearchNext', 'artefactSearchToggle'];
    const els = Object.fromEntries(ids.map((id) => [id, element()]));
    els.artefactSearchClear.hidden = true;
    els.artefactSearchPrev.disabled = els.artefactSearchNext.disabled = true;
    const controller = {renderTabStripView: vi.fn(), syncDagHighlights: vi.fn(), openOrActivateTab: vi.fn()};
    const context = {
        ...search,
        document: {getElementById: (id) => els[id]},
        window: {pywebview: {api: {search_artefacts: vi.fn().mockResolvedValue({success: true, matches})}}},
        panelControllers: {bottom: controller}, focusedPanel: 'bottom', panelFocusChosen: true,
        panelIsVisible: () => true, panelPosition: () => ({top: 0, left: 0}),
        updateFocusedPanelHighlight: vi.fn(),
        searchState: {query: '', matches: [], index: -1, token: 0, timer: null},
        searchLabels: new Set(), apiReady: true, setTimeout, clearTimeout, console,
    };
    vm.createContext(context);
    vm.runInContext(source, context);
    return {els, context, controller, api: context.window.pywebview.api};
}

const found = [
    {id: 'a1', label: 'alpha', type: 'collectra.Text', field: 'text', snippet: ''},
    {id: 'b1', label: 'beta', type: 'collectra.Text', field: 'text', snippet: ''},
    {id: 'b2', label: 'beta', type: 'collectra.Text', field: 'text', snippet: ''},
];

beforeEach(() => vi.useFakeTimers());
afterEach(() => vi.useRealTimers());

async function type(p, text) {
    p.els.artefactSearchInput.value = text;
    p.context.handleArtefactSearchInput();
    await vi.advanceTimersByTimeAsync(300);
}

it('is closed to begin with and opens with the box focused', () => {
    const p = page(found);
    expect(p.els.artefactSearchBar.classList.contains('open')).toBe(false);
    p.context.openArtefactSearch();
    expect(p.els.artefactSearchBar.classList.contains('open')).toBe(true);
    expect(p.els.artefactSearchInput.focused).toBe(true);
});

it('the magnifier button opens it, then closes it', () => {
    const p = page(found);
    p.context.toggleArtefactSearch();
    expect(p.els.artefactSearchBar.classList.contains('open')).toBe(true);
    p.context.toggleArtefactSearch();
    expect(p.els.artefactSearchBar.classList.contains('open')).toBe(false);
});

it('typing searches after a pause and shows the count, the clear button and enabled arrows', async () => {
    const p = page(found);
    p.context.openArtefactSearch();
    await type(p, 'theatre');

    expect(p.api.search_artefacts).toHaveBeenCalledWith('theatre');
    expect(p.els.artefactSearchCount.textContent).toBe('3 matches');
    expect(p.els.artefactSearchClear.hidden).toBe(false);
    expect(p.els.artefactSearchPrev.disabled).toBe(false);
    expect(p.els.artefactSearchNext.disabled).toBe(false);
    expect(p.controller.renderTabStripView).toHaveBeenCalled();
});

it('does not search on every keystroke', async () => {
    const p = page(found);
    p.els.artefactSearchInput.value = 't';
    p.context.handleArtefactSearchInput();
    p.els.artefactSearchInput.value = 'th';
    p.context.handleArtefactSearchInput();
    await vi.advanceTimersByTimeAsync(300);
    expect(p.api.search_artefacts).toHaveBeenCalledTimes(1);
    expect(p.api.search_artefacts).toHaveBeenCalledWith('th');
});

it('stepping shows "1 of 3 matches" and opens that artefact in the focused panel', async () => {
    const p = page(found);
    await type(p, 'x');
    await p.context.stepArtefactSearch(1);
    expect(p.els.artefactSearchCount.textContent).toBe('1 of 3 matches');
    expect(p.controller.openOrActivateTab).toHaveBeenLastCalledWith('a1', 'alpha');
    await p.context.stepArtefactSearch(1);
    expect(p.els.artefactSearchCount.textContent).toBe('2 of 3 matches');
    expect(p.controller.openOrActivateTab).toHaveBeenLastCalledWith('b1', 'beta');
});

it('Enter steps forward and Shift+Enter steps back', async () => {
    const p = page(found);
    await type(p, 'x');
    p.context.handleArtefactSearchKey({key: 'Enter', shiftKey: false, preventDefault() {}, target: p.els.artefactSearchInput});
    await vi.advanceTimersByTimeAsync(0);
    expect(p.els.artefactSearchCount.textContent).toBe('1 of 3 matches');
    p.context.handleArtefactSearchKey({key: 'Enter', shiftKey: true, preventDefault() {}, target: p.els.artefactSearchInput});
    await vi.advanceTimersByTimeAsync(0);
    expect(p.els.artefactSearchCount.textContent).toBe('3 of 3 matches');
});

it('the x empties the search but leaves the bar open', async () => {
    const p = page(found);
    p.context.openArtefactSearch();
    await type(p, 'theatre');
    p.context.clearArtefactSearch();
    await vi.advanceTimersByTimeAsync(0);

    expect(p.els.artefactSearchInput.value).toBe('');
    expect(p.els.artefactSearchCount.textContent).toBe('');
    expect(p.els.artefactSearchClear.hidden).toBe(true);
    expect(p.els.artefactSearchPrev.disabled).toBe(true);
    expect(p.els.artefactSearchBar.classList.contains('open')).toBe(true);
});

it('Done and Escape empty the search and close the bar', async () => {
    for (const close of [
        (p) => p.context.doneArtefactSearch(),
        (p) => p.context.handleArtefactSearchKey({key: 'Escape', target: p.els.artefactSearchInput}),
    ]) {
        const p = page(found);
        p.context.openArtefactSearch();
        await type(p, 'theatre');
        close(p);
        await vi.advanceTimersByTimeAsync(0);

        expect(p.els.artefactSearchBar.classList.contains('open')).toBe(false);
        expect(p.els.artefactSearchInput.value).toBe('');
        expect(p.els.artefactSearchCount.textContent).toBe('');
        expect(p.context.searchState.matches).toEqual([]);
    }
});

it('reports no matches when nothing is found', async () => {
    const p = page([]);
    await type(p, 'zzz');
    expect(p.els.artefactSearchCount.textContent).toBe('No matches');
    expect(p.els.artefactSearchPrev.disabled).toBe(true);
});

it('an older, slower answer never overwrites a newer one', async () => {
    const p = page();
    let resolveFirst;
    p.api.search_artefacts
        .mockImplementationOnce(() => new Promise((r) => { resolveFirst = r; }))
        .mockImplementationOnce(() => Promise.resolve({success: true, matches: found.slice(0, 1)}));
    await type(p, 'a');
    await type(p, 'ab');
    resolveFirst({success: true, matches: found});
    await vi.advanceTimersByTimeAsync(0);
    expect(p.els.artefactSearchCount.textContent).toBe('1 match');
});

describe('which panel a result opens in', () => {
    function panels({focused = 'bottom', chosen = false, visible = () => true, where = {}}) {
        const p = page(found);
        const mk = () => ({renderTabStripView: vi.fn(), syncDagHighlights: vi.fn(), openOrActivateTab: vi.fn()});
        const c = {top: mk(), panelA: mk(), bottom: mk()};
        p.context.panelControllers = c;
        p.context.focusedPanel = focused;
        p.context.panelFocusChosen = chosen;
        p.context.panelIsVisible = visible;
        p.context.panelPosition = (key) => where[key] || {top: 0, left: 0};
        return {p, c};
    }
    const layout = {top: {top: 0, left: 0}, panelA: {top: 120, left: 0}, bottom: {top: 400, left: 0}};

    it('the top panel, when you have not chosen one (the bottom one is only the starting default)', async () => {
        const {p, c} = panels({focused: 'bottom', chosen: false, where: layout});
        await type(p, 'x');
        await p.context.stepArtefactSearch(1);
        expect(c.top.openOrActivateTab).toHaveBeenCalledWith('a1', 'alpha');
        expect(c.bottom.openOrActivateTab).not.toHaveBeenCalled();
        expect(p.context.focusedPanel).toBe('top');
        expect(p.context.panelFocusChosen).toBe(false); // searching is not choosing a panel
    });

    it('the panel you chose, when you did', async () => {
        const {p, c} = panels({focused: 'panelA', chosen: true, where: layout});
        await type(p, 'x');
        await p.context.stepArtefactSearch(1);
        expect(c.panelA.openOrActivateTab).toHaveBeenCalledWith('a1', 'alpha');
        expect(c.top.openOrActivateTab).not.toHaveBeenCalled();
        expect(p.context.updateFocusedPanelHighlight).not.toHaveBeenCalled();
    });

    it('the topmost visible panel when the one you chose was deleted', async () => {
        const {p, c} = panels({focused: 'deleted', chosen: true, where: layout});
        await type(p, 'x');
        await p.context.stepArtefactSearch(1);
        expect(c.top.openOrActivateTab).toHaveBeenCalledWith('a1', 'alpha');
    });

    it('not a collapsed or hidden panel, even the top one', async () => {
        const {p, c} = panels({focused: 'bottom', chosen: false, where: layout, visible: (key) => key !== 'top'});
        await type(p, 'x');
        await p.context.stepArtefactSearch(1);
        expect(c.panelA.openOrActivateTab).toHaveBeenCalledWith('a1', 'alpha');
        expect(c.top.openOrActivateTab).not.toHaveBeenCalled();
    });

    it('nothing happens, and nothing breaks, with no panels at all', async () => {
        const p = page(found);
        p.context.panelControllers = {};
        await type(p, 'x');
        await expect(p.context.stepArtefactSearch(1)).resolves.toBeUndefined();
    });
});

it('the page decides what counts as a visible panel, and where a panel is on screen', () => {
    const fn = html.slice(html.indexOf('function panelIsVisible('), html.indexOf('\n        }\n', html.indexOf('function panelIsVisible(')));
    expect(fn).toContain('isCollapsed()');
    expect(fn).toContain('getBoundingClientRect()');
    const where = html.slice(html.indexOf('function panelPosition('), html.indexOf('\n        }\n', html.indexOf('function panelPosition(')));
    expect(where).toContain('getBoundingClientRect()');
});

describe('a panel counts as chosen only when the user chose it', () => {
    const between = (from, to) => html.slice(html.indexOf(from), html.indexOf(to, html.indexOf(from)));

    it('starts as not chosen, declared before the first tab strip is drawn', () => {
        const at = html.indexOf('let panelFocusChosen = false;');
        expect(at).toBeGreaterThan(-1);
        expect(at).toBeLessThan(html.indexOf('panelControllers.bottom.renderTabStripView();'));
    });

    it('is set by clicking in a panel, splitting, dropping a tab or using a panel\'s own buttons', () => {
        expect(between('closest(\'#imagePanelTop\')) next', '}, true);')).toContain('panelFocusChosen = true;');
        expect(between('function createPanel(direction)', '// Icon-tile popover')).toContain('panelFocusChosen = true;');
        expect(between('function openLineageFor(', 'function closeLineagePopovers()')).toContain('panelFocusChosen = true;');
        expect(html.slice(html.indexOf('focusedPanel = targetPanelKey;') - 200, html.indexOf('focusedPanel = targetPanelKey;') + 200)).toContain('panelFocusChosen = true;');
    });

    it('is cleared when the focus moves only because a panel was removed', () => {
        expect(between('function detachPanel(', 'function removePanelByKey')).toContain('panelFocusChosen = false;');
    });

    it('search uses it, and does not set it', () => {
        const step = html.slice(html.indexOf('async function stepArtefactSearch('), html.indexOf('// ── END ARTEFACT SEARCH'));
        expect(step).toContain('searchTargetPanel(Object.keys(panelControllers), panelFocusChosen ? focusedPanel : null, panelIsVisible, panelPosition)');
        expect(step).not.toContain('panelFocusChosen = true');
    });
});

describe('searching does not rewrite the saved layout', () => {
    it('redraws the tab strips without asking for the layout to be saved', async () => {
        const p = page(found);
        await type(p, 'theatre');
        expect(p.controller.renderTabStripView).toHaveBeenCalled();
        for (const call of p.controller.renderTabStripView.mock.calls) {
            expect(call[0]).toEqual({save: false});
        }
    });
});

describe('renderTabStrip', () => {
    const source = slice('function renderTabStrip(', '// Right-click on a tab');

    function run(options) {
        const stripEl = {innerHTML: 'old'};
        const context = {
            hideTabTooltip: vi.fn(), scheduleArrangementSave: vi.fn(), searchLabels: new Set(),
            escapeHtml: (x) => x, tabTypeTitle: () => '', pageNodeTypes: new Map(), resolveDagLabel: (x) => x,
        };
        vm.createContext(context);
        vm.runInContext(source, context);
        context.renderTabStrip({tabs: [], activeTabId: null, canCloseTab: () => true}, stripEl, 'bottom', options);
        return context;
    }

    it('asks for the layout to be saved by default, since tabs may have changed', () => {
        expect(run(undefined).scheduleArrangementSave).toHaveBeenCalledTimes(1);
    });

    it('does not when told the redraw is only highlights', () => {
        expect(run({save: false}).scheduleArrangementSave).not.toHaveBeenCalled();
    });
});

it('a controller passes that choice on to the tab strip', () => {
    expect(html).toContain('renderTabStripView({ save = true } = {}) {');
    expect(html).toContain('renderTabStrip(this.panel, this.tabStripEl, this.key, { save });');
});

// ---- (3) the small search-box fixes
describe('the search box keys', () => {
    it('Enter while an input method is composing text does not step to the next match', async () => {
        const p = page(found);
        await type(p, 'x');
        let prevented = false;
        p.context.handleArtefactSearchKey({key: 'Enter', isComposing: true, preventDefault() { prevented = true; }, target: p.els.artefactSearchInput});
        await vi.advanceTimersByTimeAsync(0);
        expect(prevented).toBe(false);
        expect(p.els.artefactSearchCount.textContent).toBe('3 matches'); // no step happened
    });

    it('Done and clear drop a search that was still waiting to run', async () => {
        const p = page(found);
        p.els.artefactSearchInput.value = 'abc';
        p.context.handleArtefactSearchInput();           // pending 250 ms timer
        expect(vi.getTimerCount()).toBe(1);
        p.context.doneArtefactSearch();
        expect(vi.getTimerCount()).toBe(0);
        p.els.artefactSearchInput.value = 'abc';
        p.context.handleArtefactSearchInput();
        p.context.clearArtefactSearch();
        expect(vi.getTimerCount()).toBe(0);
    });
});

describe('when there is no page to search', () => {
    it('says so, instead of "No matches"', async () => {
        const p = page(found);
        p.api.search_artefacts.mockResolvedValue({success: false, error: 'No graph loaded. Call load_yaml first.'});
        await type(p, 'loc');
        expect(p.els.artefactSearchCount.textContent).toBe('Open a page to search');
        expect(p.els.artefactSearchPrev.disabled).toBe(true);
        expect(p.els.artefactSearchNext.disabled).toBe(true);
    });

    it('finds the matches as soon as a page is open, by running the same query again', async () => {
        const p = page(found);
        p.api.search_artefacts.mockResolvedValueOnce({success: false, error: 'No graph loaded. Call load_yaml first.'});
        await type(p, 'loc');
        expect(p.els.artefactSearchCount.textContent).toBe('Open a page to search');
        p.api.search_artefacts.mockResolvedValue({success: true, matches: found});
        await p.context.rerunArtefactSearch();
        expect(p.els.artefactSearchCount.textContent).toBe('3 matches');
    });

    it('says the search failed for any other problem', async () => {
        const p = page(found);
        p.api.search_artefacts.mockResolvedValue({success: false, error: 'something else'});
        await type(p, 'loc');
        expect(p.els.artefactSearchCount.textContent).toBe('Search failed');
        p.api.search_artefacts.mockRejectedValue(new Error('bridge down'));
        await type(p, 'loc2');
        expect(p.els.artefactSearchCount.textContent).toBe('Search failed');
    });

    it('a later successful search clears the message', async () => {
        const p = page(found);
        p.api.search_artefacts.mockResolvedValueOnce({success: false, error: 'No graph loaded.'});
        await type(p, 'a');
        p.api.search_artefacts.mockResolvedValue({success: true, matches: found});
        await type(p, 'ab');
        expect(p.els.artefactSearchCount.textContent).toBe('3 matches');
    });
});
