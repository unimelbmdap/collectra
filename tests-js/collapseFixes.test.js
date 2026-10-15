const fs = require('node:fs');
const vm = require('node:vm');
const { serializeArrangement, keepOneExpanded } = require('../collectra/gui/arrangement.js');
const { searchTargetPanel } = require('../collectra/gui/artefactSearch.js');
const helpers = require('../collectra/gui/collapseToggle.js');
const html = fs.readFileSync(require.resolve('../collectra/gui/index.html'), 'utf8');
const between = (from, to) => html.slice(html.indexOf(from), html.indexOf(to, html.indexOf(from)));
const fnSource = (name, prefix = '') => {
    const at = html.indexOf(`${prefix}function ${name}(`);
    return html.slice(at, html.indexOf('\n        }\n', at)) + '\n        }\n';
};
const rule = (selector) => {
    const start = html.indexOf(`        ${selector} {`);
    expect(start, selector).toBeGreaterThan(-1);
    return html.slice(start, html.indexOf('}', start));
};

// ---------------------------------------------------------------------------------------------
describe('1. opening something never lands in a panel you cannot see', () => {
    function page({ focused, visible, where = {} }) {
        const mk = () => ({ openOrActivateTab: vi.fn().mockResolvedValue(), openInNewTab: vi.fn(), openDagKindTab: vi.fn() });
        const controllers = { top: mk(), bottom: mk(), panel2: mk() };
        const context = {
            searchTargetPanel, panelControllers: controllers, focusedPanel: focused,
            panelIsVisible: (key) => visible(key), panelPosition: (key) => where[key] || { top: 0, left: 0 },
            updateFocusedPanelHighlight: vi.fn(),
        };
        vm.createContext(context);
        vm.runInContext(['openingPanel', 'selectNode', 'openNodeInNewTab', 'openDagTab'].map((n) => fnSource(n, n === 'selectNode' ? 'async ' : '')).join('\n'), context);
        return { context, controllers };
    }
    const where = { top: { top: 0, left: 0 }, panel2: { top: 100, left: 0 }, bottom: { top: 400, left: 0 } };

    it('a list item opens in the focused panel when it can be seen', async () => {
        const { context, controllers } = page({ focused: 'bottom', visible: () => true, where });
        await context.selectNode('age');
        expect(controllers.bottom.openOrActivateTab).toHaveBeenCalledWith('age');
        expect(context.updateFocusedPanelHighlight).not.toHaveBeenCalled();
    });

    it('but in the topmost visible panel when the focused one is collapsed', async () => {
        // the focused bottom panel and the top one are both collapsed; only panel2 is open
        const { context, controllers } = page({ focused: 'bottom', visible: (k) => k === 'panel2', where });
        await context.selectNode('age');
        expect(controllers.panel2.openOrActivateTab).toHaveBeenCalledWith('age');
        expect(controllers.bottom.openOrActivateTab).not.toHaveBeenCalled();
        expect(context.focusedPanel).toBe('panel2');
    });

    it('"open in new tab" and the DAG tab follow the same rule', () => {
        const { context, controllers } = page({ focused: 'bottom', visible: (k) => k === 'panel2', where });
        context.openNodeInNewTab('age');
        context.openDagTab();
        expect(controllers.panel2.openInNewTab).toHaveBeenCalledWith('age');
        expect(controllers.panel2.openDagKindTab).toHaveBeenCalled();
        expect(controllers.bottom.openInNewTab).not.toHaveBeenCalled();
    });

    it('nothing breaks when there is no panel at all', async () => {
        const context = { searchTargetPanel, panelControllers: {}, focusedPanel: 'bottom', panelIsVisible: () => true, panelPosition: () => ({ top: 0, left: 0 }), updateFocusedPanelHighlight() {} };
        vm.createContext(context);
        vm.runInContext(fnSource('openingPanel') + fnSource('selectNode', 'async '), context);
        await expect(context.selectNode('x')).resolves.toBeUndefined();
    });
});

// ---------------------------------------------------------------------------------------------
describe('2. which panels are collapsed is remembered, and one always stays open', () => {
    const { Panel } = require('../collectra/gui/panelSystem.js');

    it('is written into the saved arrangement', () => {
        const saved = serializeArrangement({ top: new Panel(), bottom: new Panel() }, 'bottom', (x) => x, { collapsed: ['top'] });
        expect(saved.collapsed).toEqual(['top']);
    });

    it('is left out when the caller does not say (an old save then behaves as before)', () => {
        expect(serializeArrangement({ top: new Panel() }, 'top', (x) => x)).not.toHaveProperty('collapsed');
    });

    describe('keepOneExpanded', () => {
        it('has nothing to do while any panel is open, or when there are no panels', () => {
            expect(keepOneExpanded(['top', 'bottom'], (k) => k === 'top')).toBe(null);
            expect(keepOneExpanded([], () => true)).toBe(null);
        });

        it('opens a panel when all of them are collapsed, preferring one that is not the top DAG panel', () => {
            expect(keepOneExpanded(['top', 'bottom', 'panel2'], () => true)).toBe('bottom');
            expect(keepOneExpanded(['top'], () => true)).toBe('top');
        });
    });

    it('the page saves it, and saves again whenever a panel is collapsed or expanded', () => {
        expect(between('function currentArrangement()', 'function scheduleArrangementSave')).toContain('collapsed: currentCollapsedKeys()');
        expect(fnSource('applyPanelCollapse')).toContain('scheduleArrangementSave();');
    });

    it('and puts it back after the panels are rebuilt, then makes sure one panel is open', () => {
        const restore = between('async function restoreArrangement(', 'END REMEMBERED ARRANGEMENT');
        expect(restore.indexOf('applyPanelSizes(saved.sizes, keyMap);')).toBeLessThan(restore.indexOf('saved.collapsed'));
        expect(restore).toContain('Array.isArray(saved.collapsed)');
        expect(restore).toContain('keepOneExpanded(');
    });

    it('currentCollapsedKeys lists the collapsed panels', () => {
        const context = { panelCollapseRegistry: { top: { isCollapsed: () => true }, bottom: { isCollapsed: () => false }, p2: { isCollapsed: () => true } } };
        vm.createContext(context);
        vm.runInContext(fnSource('currentCollapsedKeys'), context);
        expect(context.currentCollapsedKeys()).toEqual(['top', 'p2']);
    });
});

// ---------------------------------------------------------------------------------------------
describe('3. a collapsed bar follows its panel when the layout around it changes', () => {
    function panel(parentClass, classes) {
        const set = new Set(classes);
        return {
            parentElement: { classList: { contains: (c) => c === parentClass } },
            previousElementSibling: null,
            classList: { toggle: (c, on) => (on ? set.add(c) : set.delete(c)), contains: (c) => set.has(c) },
            querySelector: () => null,
            set,
        };
    }

    function setup(entries) {
        const roots = Object.fromEntries(Object.entries(entries).map(([k, e]) => [k, e.root]));
        const context = {
            ...helpers,
            scheduleArrangementSave: vi.fn(),
            panelCollapseRegistry: Object.fromEntries(Object.entries(entries).map(([k, e]) => [k, { isCollapsed: () => e.collapsed }])),
            getPanelRootEl: (key) => roots[key],
        };
        vm.createContext(context);
        vm.runInContext(fnSource('applyPanelCollapse') + fnSource('refreshCollapsedOrientations'), context);
        return context;
    }

    it('a bar that ended up beside another panel becomes a vertical one', () => {
        const root = panel('split-row', ['collapsed', 'collapsed-h']); // it was collapsed while stacked
        setup({ p: { root, collapsed: true } }).refreshCollapsedOrientations();
        expect([...root.set].sort()).toEqual(['collapsed', 'collapsed-v']);
    });

    it('and the other way round', () => {
        const root = panel('split-column', ['collapsed', 'collapsed-v']);
        setup({ p: { root, collapsed: true } }).refreshCollapsedOrientations();
        expect([...root.set].sort()).toEqual(['collapsed', 'collapsed-h']);
    });

    it('open panels are left alone', () => {
        const root = panel('split-row', []);
        setup({ p: { root, collapsed: false } }).refreshCollapsedOrientations();
        expect(root.set.size).toBe(0);
    });

    it('is done after a panel is added and after one is removed', () => {
        expect(between('function createPanel(direction)', '// Icon-tile popover')).toContain('refreshCollapsedOrientations();');
        expect(fnSource('detachPanel')).toContain('refreshCollapsedOrientations();');
    });
});

// ---------------------------------------------------------------------------------------------
describe('4. disabled buttons stay transparent', () => {
    // The page-wide `button:disabled` paints a dark box; these rules have to out-rank it.
    it.each(['.panel-collapse-btn:disabled', '.panel-close-btn:disabled', '.node-action-btn:disabled'])('%s', (selector) => {
        expect(rule(selector)).toContain('background: transparent;');
    });
});

describe('state the start-up code reads is declared before it runs', () => {
    // Collapsing a panel now asks to save the layout, and it runs while the page loads.
    it('the save state comes before the first collapse at start-up', () => {
        const first = html.indexOf('applyDagCollapsedState(); // sync DOM');
        expect(first).toBeGreaterThan(-1);
        for (const name of ['arrangementReady', 'arrangementSaveTimer', 'layoutOps', 'replayingLayout', 'searchLabels', 'searchState']) {
            const declared = html.indexOf(`let ${name}`);
            expect(declared, name).toBeGreaterThan(-1);
            expect(declared, name).toBeLessThan(first);
        }
    });
});
