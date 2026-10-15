const { Tab, Panel } = require('../collectra/gui/panelSystem.js');
const { serializeArrangement, restorePanelTabs, replayLayout, sizeToPercent } = require('../collectra/gui/arrangement.js');

const labelOf = (nodeId) => nodeId.replace(/\d+$/, '');
const humanize = (label) => label.split('_').map((w) => w[0].toUpperCase() + w.slice(1)).join(' ');

function panelWith(tabs, activeId) {
    const panel = new Panel();
    tabs.forEach((t) => panel.addTab(new Tab(t)));
    if (activeId) panel.activateTab(activeId);
    return panel;
}

describe('serializeArrangement', () => {
    it('records each panel as a dictionary of its tabs and which one is active', () => {
        const bottom = panelWith([
            { id: 'a', nodeId: 'specimen_sheet', groupKey: 'specimen_sheet', kind: 'node' },
            { id: 'b', nodeId: 'primary_label_unoriented1', groupKey: 'primary_label_unoriented', kind: 'node' },
        ], 'b');
        const top = panelWith([{ id: 'd', nodeId: null, label: 'DAG', kind: 'dag' }], 'd');

        expect(serializeArrangement({ top, bottom }, 'bottom', labelOf)).toEqual({
            version: 1,
            focused: 'bottom',
            layout: [],
            sizes: {},
            panels: {
                top: { active: 0, tabs: [{ kind: 'dag' }] },
                bottom: {
                    active: 1,
                    tabs: [
                        { kind: 'node', group: 'specimen_sheet' },
                        { kind: 'node', group: 'primary_label_unoriented' },
                    ],
                },
            },
        });
    });

    it('stores a blank tab by its label so it can come back', () => {
        const panel = panelWith([{ id: 'a', nodeId: null, groupKey: 'locality', kind: 'node' }], 'a');
        expect(serializeArrangement({ bottom: panel }, 'bottom', labelOf).panels.bottom.tabs)
            .toEqual([{ kind: 'node', group: 'locality' }]);
    });

    it('falls back to the label of the shown instance for a tab without a group', () => {
        const panel = panelWith([{ id: 'a', nodeId: 'locality3', kind: 'node' }], 'a');
        expect(serializeArrangement({ bottom: panel }, 'bottom', labelOf).panels.bottom.tabs)
            .toEqual([{ kind: 'node', group: 'locality' }]);
    });

    it('records an empty panel with no active tab', () => {
        expect(serializeArrangement({ bottom: new Panel() }, 'bottom', labelOf).panels.bottom)
            .toEqual({ active: 0, tabs: [] });
    });
});

describe('restorePanelTabs', () => {
    const spec = {
        active: 1,
        tabs: [{ kind: 'dag' }, { kind: 'node', group: 'primary_label_unoriented' }, { kind: 'node', group: 'locality' }],
    };
    let n;
    const makeId = () => 'tab' + ++n;
    const options = (map) => ({ makeId, instancesFor: (g) => map[g] || [], labelFor: humanize });
    beforeEach(() => { n = 0; });

    it('opens every saved tab, showing the first instance this page has', () => {
        const panel = new Panel();
        restorePanelTabs(panel, spec, options({ primary_label_unoriented: ['x-1', 'x-2'], locality: ['loc-1'] }));

        expect(panel.tabs.map((t) => [t.kind, t.groupKey, t.nodeId])).toEqual([
            ['dag', null, null],
            ['node', 'primary_label_unoriented', 'x-1'],
            ['node', 'locality', 'loc-1'],
        ]);
        expect(panel.tabs[1].label).toBe('Primary Label Unoriented');
        expect(panel.tabs[0].label).toBe('DAG');
    });

    it('leaves a tab blank when this page has no instance of its node', () => {
        const panel = new Panel();
        restorePanelTabs(panel, spec, options({ locality: ['loc-1'] }));

        const blank = panel.tabs[1];
        expect(blank.groupKey).toBe('primary_label_unoriented');
        expect(blank.nodeId).toBe(null);
        expect(panel.tabs).toHaveLength(3);
    });

    it('re-activates the tab that was active', () => {
        const panel = new Panel();
        restorePanelTabs(panel, spec, options({}));
        expect(panel.getActiveTab().groupKey).toBe('primary_label_unoriented');
    });

    it('replaces whatever tabs the panel had', () => {
        const panel = panelWith([{ id: 'old', nodeId: 'z', kind: 'node' }], 'old');
        restorePanelTabs(panel, spec, options({}));
        expect(panel.getTab('old')).toBe(null);
    });

    it('leaves the panel as it is when nothing was saved for it', () => {
        const panel = panelWith([{ id: 'keep', nodeId: 'z', kind: 'node' }], 'keep');
        restorePanelTabs(panel, null, options({}));
        restorePanelTabs(panel, undefined, options({}));
        expect(panel.getTab('keep')).not.toBe(null);
    });

    it('empties the panel when it was saved with every tab closed', () => {
        const panel = panelWith([{ id: 'old', nodeId: 'z', kind: 'node' }], 'old');
        restorePanelTabs(panel, { active: 0, tabs: [] }, options({}));
        expect(panel.tabs).toEqual([]);
        expect(panel.activeTabId).toBe(null);
    });

    it('does not open the same node twice', () => {
        const panel = new Panel();
        restorePanelTabs(panel, { active: 0, tabs: [{ kind: 'node', group: 'a' }, { kind: 'node', group: 'a' }] }, options({}));
        expect(panel.tabs).toHaveLength(1);
    });
});

describe('serializeArrangement with the layout', () => {
    it('stores the split log and panel sizes alongside the tabs', () => {
        const layout = [{ op: 'split', from: 'bottom', direction: 'right', key: 'panel2' }];
        const sizes = { bottom: '0 0 300px' };
        const result = serializeArrangement({ bottom: new Panel() }, 'bottom', labelOf, { layout, sizes });
        expect(result.layout).toEqual(layout);
        expect(result.sizes).toEqual(sizes);
    });

    it('defaults both to empty', () => {
        const result = serializeArrangement({ bottom: new Panel() }, 'bottom', labelOf);
        expect(result.layout).toEqual([]);
        expect(result.sizes).toEqual({});
    });
});

describe('replayLayout', () => {
    function fakePage(existing = ['top', 'bottom']) {
        const keys = new Set(existing);
        let next = 2;
        const calls = [];
        return {
            calls,
            keys,
            has: (key) => keys.has(key),
            split: (from, direction) => {
                const key = 'panel' + next++;
                keys.add(key);
                calls.push(['split', from, direction, key]);
                return key;
            },
            remove: (key) => { keys.delete(key); calls.push(['remove', key]); },
        };
    }

    it('redoes the splits and closes in the order they happened', () => {
        const page = fakePage();
        replayLayout([
            { op: 'split', from: 'bottom', direction: 'right', key: 'panel2' },
            { op: 'split', from: 'panel2', direction: 'below', key: 'panel3' },
            { op: 'remove', key: 'panel2' },
        ], page);
        expect(page.calls).toEqual([
            ['split', 'bottom', 'right', 'panel2'],
            ['split', 'panel2', 'below', 'panel3'],
            ['remove', 'panel2'],
        ]);
        expect([...page.keys].sort()).toEqual(['bottom', 'panel3', 'top']);
    });

    it('maps saved panel names onto the panels the page really created', () => {
        const page = fakePage();
        page.split = (from, direction) => { page.keys.add('panel9'); page.calls.push(['split', from, direction]); return 'panel9'; };
        const map = replayLayout([
            { op: 'split', from: 'bottom', direction: 'left', key: 'panel2' },
            { op: 'split', from: 'panel2', direction: 'above', key: 'panel3' },
        ], page);
        expect(map.panel2).toBe('panel9');
        expect(page.calls[1]).toEqual(['split', 'panel9', 'above']);
    });

    it('returns an identity map for the panels that always exist', () => {
        const map = replayLayout([], fakePage());
        expect(map).toEqual({ top: 'top', bottom: 'bottom' });
    });

    it('skips a split from a panel that is not there, and a close of one that is not', () => {
        const page = fakePage();
        replayLayout([
            { op: 'split', from: 'panel7', direction: 'right', key: 'panel8' },
            { op: 'remove', key: 'panel8' },
        ], page);
        expect(page.calls).toEqual([]);
    });

    it('does nothing for a missing log', () => {
        const page = fakePage();
        expect(replayLayout(undefined, page)).toEqual({ top: 'top', bottom: 'bottom' });
        expect(page.calls).toEqual([]);
    });
});

describe('wiring in the page', () => {
    const html = require('node:fs').readFileSync(require.resolve('../collectra/gui/index.html'), 'utf8');

    it('loads the module and restores before the first page opens', () => {
        expect(html).toContain('<script src="arrangement.js"></script>');
        const ready = html.slice(html.indexOf("addEventListener('pywebviewready'"));
        expect(ready.indexOf('await restoreArrangement();')).toBeGreaterThan(-1);
        expect(ready.indexOf('await restoreArrangement();')).toBeLessThan(ready.indexOf('window.pywebview.api.get_initial_items()'));
    });

    it('does not save until the saved arrangement has been restored', () => {
        expect(html).toContain('if (!arrangementReady || !apiReady) return;');
        expect(html).toContain('arrangementReady = true;');
    });

    it('saves whenever a tab strip is redrawn or the focused panel changes', () => {
        const strip = html.slice(html.indexOf('function renderTabStrip('), html.indexOf('function activateTabInPanel'));
        expect(strip).toContain('scheduleArrangementSave();');
        const focus = html.slice(html.indexOf('function updateFocusedPanelHighlight'), html.indexOf('function updateFocusedPanelHighlight') + 600);
        expect(focus).toContain('scheduleArrangementSave();');
    });
});

describe('extra panels in the page', () => {
    const html = require('node:fs').readFileSync(require.resolve('../collectra/gui/index.html'), 'utf8');
    const between = (from, to) => html.slice(html.indexOf(from), html.indexOf(to, html.indexOf(from)));

    it('a page switch no longer throws the extra panels away', () => {
        expect(between('function clearPageView()', 'clearImage();')).not.toContain('collapseToSinglePanel');
        expect(html).not.toContain('function collapseToSinglePanel');
    });

    it('records each split and each user close in the layout log', () => {
        expect(between('function createPanel(direction)', '// Icon-tile popover')).toContain("recordLayoutOp({op: 'split', from: sourceKey, direction, key});");
        expect(between('function removePanelByKey', 'wireTabDropOnPanel(')).toContain("recordLayoutOp({op: 'remove', key});");
    });

    it('saves after a panel is resized', () => {
        expect(between('function wireGenericResizeHandle', '// Splits the focused panel')).toContain('scheduleArrangementSave();');
    });

    it('stores the log and the panel sizes with the arrangement', () => {
        expect(between('function currentArrangement()', 'function scheduleArrangementSave')).toContain('{layout: layoutOps, sizes: currentPanelSizes(), collapsed: currentCollapsedKeys()}');
    });

    it('rebuilds the panels before restoring tabs, then puts sizes and focus back', () => {
        const restore = between('async function restoreArrangement()', 'END REMEMBERED ARRANGEMENT');
        expect(restore).toContain('replayLayout(saved.layout, {');
        expect(restore.indexOf('replayLayout(')).toBeLessThan(restore.indexOf('restorePanelTabs('));
        expect(restore).toContain('applyPanelSizes(saved.sizes, keyMap);');
        expect(restore).toContain('keyMap[saved.focused]');
    });
});

describe('startup order', () => {
    const html = require('node:fs').readFileSync(require.resolve('../collectra/gui/index.html'), 'utf8');

    // The tab strips are drawn while the page script is still loading, and
    // drawing one can schedule a save, so the state it reads must already exist.
    it('declares the save state before the first tab strip is drawn', () => {
        const firstDraw = html.indexOf('panelControllers.bottom.renderTabStripView();');
        for (const name of ['arrangementReady', 'arrangementSaveTimer', 'layoutOps', 'replayingLayout']) {
            const declared = html.indexOf(`let ${name}`);
            expect(declared, name).toBeGreaterThan(-1);
            expect(declared, name).toBeLessThan(firstDraw);
        }
    });
});

describe('sizeToPercent', () => {
    it('turns a saved pixel size into a share of the container', () => {
        expect(sizeToPercent('0 0 344.5px', 689)).toBe('0 1 50.00%');
        expect(sizeToPercent('0 0 704px', 1000)).toBe('0 1 70.40%');
    });

    it('keeps a percentage, making it shrinkable', () => {
        expect(sizeToPercent('0 0 45.50%', 1000)).toBe('0 1 45.50%');
        expect(sizeToPercent('0 1 30.00%', 1000)).toBe('0 1 30.00%');
    });

    it('never lets one side take more than 90% or less than 10%', () => {
        expect(sizeToPercent('0 0 10px', 1000)).toBe('0 1 10.00%');
        expect(sizeToPercent('0 0 990px', 1000)).toBe('0 1 90.00%');
        expect(sizeToPercent('0 0 99%', 1000)).toBe('0 1 90.00%');
    });

    it('is null for a value it cannot use, or an unknown container size', () => {
        expect(sizeToPercent('calc(evil)', 1000)).toBe(null);
        expect(sizeToPercent('0 0 300px', 0)).toBe(null);
        expect(sizeToPercent('', 1000)).toBe(null);
        expect(sizeToPercent(undefined, 1000)).toBe(null);
    });
});

describe('restoring sizes in the page', () => {
    const vm = require('node:vm');
    const html = require('node:fs').readFileSync(require.resolve('../collectra/gui/index.html'), 'utf8');
    const start = html.indexOf('function applyPanelSizes(');
    const source = html.slice(start, html.indexOf('\n        }\n', start) + 11);

    function page() {
        const column = { classList: { contains: (c) => c === 'split-column' }, getBoundingClientRect: () => ({ width: 900, height: 600 }) };
        const row = { classList: { contains: (c) => c === 'split-row' }, getBoundingClientRect: () => ({ width: 1000, height: 500 }) };
        const mk = (parent) => ({ style: {}, parentElement: parent });
        const els = { a: mk(column), b: mk(row), c: mk(null) };
        const context = { sizeToPercent, panelControllers: { a: {}, b: {}, c: {} }, getPanelRootEl: (k) => els[k] };
        vm.createContext(context);
        vm.runInContext(source, context);
        return { els, context };
    }

    it('converts an old pixel size into a share of the panel\'s own container', () => {
        const { els, context } = page();
        context.applyPanelSizes({ a: '0 0 300px', b: '0 0 250px' }, { a: 'a', b: 'b' });
        expect(els.a.style.flex).toBe('0 1 50.00%'); // 300 of the 600px column height
        expect(els.b.style.flex).toBe('0 1 25.00%'); // 250 of the 1000px row width
    });

    it('applies a percentage directly', () => {
        const { els, context } = page();
        context.applyPanelSizes({ a: '0 0 40.00%' }, { a: 'a' });
        expect(els.a.style.flex).toBe('0 1 40.00%');
    });

    it('leaves a panel with no wrapper, or an unusable size, as it is', () => {
        const { els, context } = page();
        context.applyPanelSizes({ c: '0 0 300px', a: 'junk' }, { a: 'a', c: 'c' });
        expect(els.c.style.flex).toBeUndefined();
        expect(els.a.style.flex).toBeUndefined();
    });
});
