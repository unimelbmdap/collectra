const fs = require('node:fs');
const vm = require('node:vm');
const helpers = require('../collectra/gui/collapseToggle.js');
const html = fs.readFileSync(require.resolve('../collectra/gui/index.html'), 'utf8');
const between = (from, to) => html.slice(html.indexOf(from), html.indexOf(to, html.indexOf(from)));
const rule = (selector) => {
    const start = html.indexOf(`        ${selector} {`);
    expect(start, selector).toBeGreaterThan(-1);
    return html.slice(start, html.indexOf('}', start));
};

describe('panelCollapseOrientation', () => {
    it('is vertical for a panel beside another one, horizontal otherwise', () => {
        expect(helpers.panelCollapseOrientation({ classList: { contains: (c) => c === 'split-row' } })).toBe('vertical');
        expect(helpers.panelCollapseOrientation({ classList: { contains: (c) => c === 'split-column' } })).toBe('horizontal');
        expect(helpers.panelCollapseOrientation({ classList: { contains: () => false } })).toBe('horizontal');
        expect(helpers.panelCollapseOrientation(null)).toBe('horizontal');
    });
});

describe('collapseChevron is the VS Code twisty', () => {
    it('points down while the panel is open and right once it is collapsed, whichever way it is laid out', () => {
        expect(helpers.collapseChevron(false)).toBe('fa-chevron-down');
        expect(helpers.collapseChevron(true)).toBe('fa-chevron-right');
    });
});

describe('collapseButtonState', () => {
    it('collapses an open panel, and says so', () => {
        expect(helpers.collapseButtonState(false, false)).toEqual({ disabled: false, title: 'Collapse panel' });
    });

    it('expands a collapsed panel, and says so', () => {
        expect(helpers.collapseButtonState(true, false)).toEqual({ disabled: false, title: 'Expand panel' });
    });

    it('is greyed out, with the reason, on the only open panel, which has nowhere to give its space', () => {
        expect(helpers.collapseButtonState(false, true)).toEqual({ disabled: true, title: "Can't collapse the only open panel" });
    });
});

describe('applyPanelCollapse', () => {
    const source = between('function applyPanelCollapse(', '\n        }\n') + '\n        }\n';

    function panel({ parentClass, first = false }) {
        const classes = new Set();
        const icon = { className: '' };
        const button = { title: '', attrs: {}, setAttribute(k, v) { this.attrs[k] = v; }, querySelector: () => icon };
        const root = {
            parentElement: { classList: { contains: (c) => c === parentClass } },
            previousElementSibling: first ? null : {},
            classList: {
                toggle: (c, on) => { on ? classes.add(c) : classes.delete(c); },
                contains: (c) => classes.has(c),
            },
            querySelector: (sel) => (sel === '.panel-collapse-btn' ? button : null),
        };
        const context = { ...helpers, scheduleArrangementSave: vi.fn() };
        vm.createContext(context);
        vm.runInContext(source, context);
        return { root, classes, icon, button, apply: context.applyPanelCollapse };
    }

    it('a stacked panel collapses to a horizontal bar', () => {
        const p = panel({ parentClass: 'split-column' });
        p.apply(p.root, true);
        expect([...p.classes].sort()).toEqual(['collapsed', 'collapsed-h']);
        expect(p.icon.className).toBe('fa fa-chevron-right');
        expect(p.button.title).toBe('Expand panel');
    });

    it('a side-by-side panel collapses to a vertical bar', () => {
        const p = panel({ parentClass: 'split-row', first: true });
        p.apply(p.root, true);
        expect([...p.classes].sort()).toEqual(['collapsed', 'collapsed-v']);
        expect(p.icon.className).toBe('fa fa-chevron-right'); // same twisty, any layout
    });

    it('expanding clears every collapsed class and says Collapse again', () => {
        const p = panel({ parentClass: 'split-column' });
        p.apply(p.root, true);
        p.apply(p.root, false);
        expect(p.classes.size).toBe(0);
        expect(p.icon.className).toBe('fa fa-chevron-down');
        expect(p.button.title).toBe('Collapse panel');
        expect(p.button.attrs['aria-label']).toBe('Collapse panel');
    });
});

describe('the button sits at the left of the tab row', () => {
    it.each([
        ['the DAG panel at the top (it starts collapsed)', 'id="dagPanelTabStrip"', 'fa-chevron-right'],
        ['the bottom panel', 'id="panelTabStrip"', 'fa-chevron-down'],
        ['a new panel', '<div class="tab-strip"></div>', 'fa-chevron-down'],
    ])('in %s, before the first tab', (_name, marker, chevron) => {
        const at = html.indexOf(marker);
        const row = html.slice(html.lastIndexOf('<div class="tab-strip-row">', at), at);
        expect(row).toContain('class="panel-collapse-btn"');
        expect(row).toContain(chevron);
    });

    it('the floating corner buttons are gone', () => {
        expect(html).not.toContain('bottomPanelCollapseToggle');
        expect(html).not.toContain('bottom-panel-collapse-toggle');
        expect(html).not.toContain('new-panel-collapse-toggle');
        expect(html).not.toContain('dagCollapseToggle');
        expect(html).not.toContain('dag-collapse-toggle');
    });

    it('is a narrow icon button, centred with the tabs, in the colour it always had', () => {
        const btn = rule('.panel-collapse-btn');
        expect(btn).toContain('width: 14px;');
        expect(btn).toContain('height: 22px;');
        expect(btn).toContain('margin-left: 4px;');
        expect(btn).toContain('align-self: center;');
        expect(btn).toContain('color: var(--app-fg-muted, #8a9698);');
        expect(rule('.panel-close-btn')).toContain('align-self: center;');
    });

    it('has the first tab right next to it, with no empty space between', () => {
        expect(rule('.tab-strip-row')).toContain('gap: 2px;');
        expect(rule('.panel-collapse-btn + .tab-strip')).toContain('padding-left: 2px;');
    });

    it('a collapsed bar keeps its tabs on one row, so the button is never pushed out of view', () => {
        const strip = rule('.collapsed-h .tab-strip');
        expect(strip).toContain('flex-wrap: nowrap;');
        expect(strip).toContain('overflow: hidden;');
    });
});

describe('a collapsed panel shrinks to a narrow bar and stays where it was', () => {
    const hList = '.image-panel-top.collapsed-h,\n        .image-panel-bottom.collapsed-h,\n        .dynamic-panel.collapsed-h';
    const vList = '.image-panel-top.collapsed-v,\n        .image-panel-bottom.collapsed-v,\n        .dynamic-panel.collapsed-v';

    it('a stacked panel keeps only its tab row', () => {
        const h = rule(hList);
        expect(h).toContain('flex: 0 0 var(--collapsed-bar) !important;');
        expect(h).toContain('min-height: var(--collapsed-bar) !important;');
    });

    it('a side-by-side panel becomes a thin vertical bar', () => {
        const v = rule(vList);
        expect(v).toContain('flex: 0 0 var(--collapsed-bar) !important;');
        expect(v).toContain('min-width: var(--collapsed-bar) !important;');
        expect(rule('.collapsed-v .tab-strip-row')).toContain('flex-direction: column;');
        expect(rule('.collapsed-v .tab-strip')).toContain('display: none;');
    });

    it('nothing is moved: no panel is re-ordered or pushed to an edge', () => {
        expect(html).not.toContain('order: 9;');
        expect(html).not.toContain('order: -1;');
        expect(rule(hList)).not.toContain('margin-top: auto');
    });

    it('a stacked bar has its line on the side facing the next panel', () => {
        expect(rule('.collapsed-h .tab-strip-row')).toContain('border-bottom: 1px solid var(--app-divider');
        // the last panel has nothing after it, so its line goes on top instead
        expect(rule('.collapsed-h:last-child')).toContain('border-top: 1px solid var(--app-divider');
        expect(rule('.collapsed-h:last-child .tab-strip-row')).toContain('border-bottom: none;');
    });

    it('a side-by-side bar does the same sideways', () => {
        expect(rule('.collapsed-v .tab-strip-row')).toContain('border-right: 1px solid var(--app-divider');
        const last = rule('.collapsed-v:last-child .tab-strip-row');
        expect(last).toContain('border-right: none;');
        expect(last).toContain('border-left: 1px solid var(--app-divider');
    });

    it('the divider next to a collapsed bar is hidden, since the bar has its own edge', () => {
        const start = html.indexOf('.collapsed + .panel-resize-handle');
        expect(start).toBeGreaterThan(-1);
        const hidden = html.slice(start, html.indexOf('}', start));
        for (const sel of ['.collapsed + .image-panel-resize-handle', '.panel-resize-handle:has(+ .collapsed)', '.image-panel-resize-handle:has(+ .collapsed)']) {
            expect(hidden).toContain(sel);
        }
        expect(hidden).toContain('display: none;');
    });

    it('everything under the tab row is hidden', () => {
        expect(rule('.collapsed .tab-strip-row ~ *')).toContain('display: none !important;');
    });

    it('the neighbour takes the freed space, in a split and in the top/bottom pair', () => {
        const split = rule('.split-wrapper:has(> .collapsed) > :not(.collapsed):not(.panel-resize-handle):not(.image-panel-resize-handle)');
        expect(split).toContain('flex: 1 1 0 !important;');
        const main = rule('.image-panel:has(> .collapsed) > :not(.collapsed):not(.image-panel-resize-handle)');
        expect(main).toContain('flex: 1 1 0 !important;');
    });

    it('one bar size is shared', () => {
        expect(html).toContain('--collapsed-bar: 34px;');
    });
});

describe('the last open panel cannot be collapsed, and says why', () => {
    const vm = require('node:vm');
    const only = between('function isOnlyExpandedPanel(', '\n        }\n') + '\n        }\n';
    const update = between('function updatePanelCloseButtons(', '\n        }\n') + '\n        }\n';

    function page(collapsedKeys) {
        const mk = () => ({ disabled: false, title: '', setAttribute(k, v) { this.attrs = { ...(this.attrs || {}), [k]: v }; } });
        const keys = ['top', 'bottom', 'panel2'];
        const buttons = Object.fromEntries(keys.map((k) => [k, { close: mk(), collapse: mk() }]));
        const registry = Object.fromEntries(keys.map((k) => [k, {
            isCollapsed: () => collapsedKeys.includes(k),
            closeBtn: () => buttons[k].close,
            collapseBtn: () => buttons[k].collapse,
        }]));
        const context = { ...helpers, panelCollapseRegistry: registry, panelControllers: Object.fromEntries(keys.map((k) => [k, {}])) };
        vm.createContext(context);
        vm.runInContext(only + update, context);
        context.updatePanelCloseButtons();
        return buttons;
    }

    it('with several panels open, every chevron works', () => {
        const b = page([]);
        for (const k of ['top', 'bottom', 'panel2']) {
            expect(b[k].collapse.disabled).toBe(false);
            expect(b[k].collapse.title).toBe('Collapse panel');
        }
    });

    it('when only one is still open, its chevron is greyed out with the reason', () => {
        const b = page(['top', 'panel2']);
        expect(b.bottom.collapse.disabled).toBe(true);
        expect(b.bottom.collapse.title).toBe("Can't collapse the only open panel");
        expect(b.bottom.collapse.attrs['aria-label']).toBe("Can't collapse the only open panel");
    });

    it('collapsed panels can always be expanded again', () => {
        const b = page(['top', 'panel2']);
        expect(b.top.collapse.disabled).toBe(false);
        expect(b.top.collapse.title).toBe('Expand panel');
    });

    it('every kind of panel registers its button', () => {
        const registry = between('const panelCollapseRegistry = {', '\n        };');
        expect(registry).toContain("collapseBtn: () => document.getElementById('dagPanelCollapseBtn')");
        expect(registry).toContain("collapseBtn: () => document.getElementById('bottomPanelCollapseBtn')");
        expect(html).toContain("collapseBtn: () => rootEl.querySelector('.panel-collapse-btn')");
    });

    it('a greyed-out chevron looks disabled', () => {
        expect(rule('.panel-collapse-btn:disabled')).toContain('opacity: 0.35;');
    });
});

describe('expanding', () => {
    it('clicking a tab in a collapsed panel expands it, as in VS Code', () => {
        const fn = between('function activateTabInPanel(', 'function closeTabInPanel');
        expect(fn).toContain('expandPanelIfCollapsed(panelKey);');
        expect(html).toContain('function expandPanelIfCollapsed(');
    });

    it('the DAG panel at the top uses the same collapse as every other panel', () => {
        const fn = between('function applyDagCollapsedState(', '\n        }\n');
        expect(fn).toContain("applyPanelCollapse(document.getElementById('imagePanelTop'), dagRegion.collapsed);");
        expect(fn).toContain('cy.fit(undefined, DAG_FIT_PADDING)'); // the graph re-fits when it opens
        expect(fn).not.toContain('0 0 0');
        expect(html).toContain('onclick="toggleDagCollapse()"');
        expect(html).not.toContain('dagPrevFlex');
    });

    it('the old rules that hid the DAG panel completely are gone', () => {
        expect(html).not.toContain('.image-panel-top.collapsed > *');
        expect(html).not.toContain('.image-panel-top.collapsed {');
    });

    it('both panel kinds collapse through the shared function', () => {
        expect(between('function applyBottomPanelCollapsedState(', '\n        }\n')).toContain("applyPanelCollapse(document.getElementById('imagePanelBottom'), bottomPanelRegion.collapsed);");
        expect(between('function applyDynamicPanelCollapsedState(', '\n        }\n')).toContain('applyPanelCollapse(controller.rootEl, collapsed);');
    });
});

describe('every kind of panel gets the thin bar, not just the ones added with Customize Layout', () => {
    // applyPanelCollapse puts collapsed-v on whichever panel is collapsed, so each of
    // them needs a rule — the original top and bottom panels too, once they are split.
    it.each(['.image-panel-top', '.image-panel-bottom', '.dynamic-panel'])('%s', (kind) => {
        const list = html.slice(html.indexOf(`        ${'.image-panel-top.collapsed-v'}`), html.indexOf('{', html.indexOf('.image-panel-top.collapsed-v')));
        expect(list).toContain(`${kind}.collapsed-v`);
    });
});
