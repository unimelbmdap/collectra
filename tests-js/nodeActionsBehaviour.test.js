// Runs the page's real row-building, popover and toolbar code against stand-in elements.
const fs = require('node:fs');
const vm = require('node:vm');
const { nodeActionButtons, pagerState } = require('../collectra/gui/instancePager.js');
const { IconPopover } = require('../collectra/gui/iconPopover.js');
const html = fs.readFileSync(require.resolve('../collectra/gui/index.html'), 'utf8');

const slice = (from, to) => html.slice(html.indexOf(from), html.indexOf(to, html.indexOf(from)));

function fakeElement(tag = 'span') {
    const listeners = {};
    return {
        tag, children: [], className: '', innerHTML: '', textContent: '', title: '', disabled: false, hidden: false,
        attrs: {}, style: {}, dataset: {}, listeners,
        setAttribute(k, v) { this.attrs[k] = v; },
        addEventListener(name, fn) { listeners[name] = fn; },
        click() { listeners.click && listeners.click(); },
        replaceChildren(...kids) { this.children = kids; },
        appendChild(kid) { this.children.push(kid); },
        querySelector() { return null; },
    };
}

describe('the row under a tab', () => {
    const source = slice('renderPager() {', 'reconcileTabs(options)');

    function setup({ tab, textSave = null, lineage = { ancestors: true, descendants: true }, instances = [tab && tab.nodeId] }) {
        const pagerEl = fakeElement();
        const calls = { del: [], lineage: [] };
        const controller = {
            el: { headerType: { parentElement: { querySelector: () => pagerEl } } },
            panel: { getActiveTab: () => tab },
            instancesForTab: () => instances,
            selectInstance: vi.fn(),
            textSave,
        };
        const context = {
            pagerState, nodeActionButtons,
            NODE_ACTION_ICONS: { delete: 'D', ancestors: 'A', descendants: 'E' },
            document: { createElement: fakeElement },
            lineageAvailability: () => lineage,
            deleteNodeWithConfirm: (...a) => calls.del.push(a),
            openLineageFor: (...a) => calls.lineage.push(a),
        };
        vm.createContext(context);
        vm.runInContext(`var holder = { ${source} };`, context);
        context.holder.renderPager.call(controller);
        return { pagerEl, controller, calls };
    }

    const textTab = { kind: 'node', nodeId: 'locality-1', groupKey: 'locality' };
    const buttons = (pagerEl) => pagerEl.children.filter((c) => c.className.startsWith('node-action-btn'));

    it('shows the id and artefact actions for editable text', () => {
        const { pagerEl } = setup({ tab: textTab, textSave: { onSave: vi.fn() } });
        expect(pagerEl.hidden).toBe(false);
        expect(pagerEl.children.find((c) => c.className === 'pager-name').textContent).toBe('locality-1');
        expect(buttons(pagerEl).map((b) => b.title)).toEqual(['Delete artefact', 'Ancestors', 'Descendants']);
    });

    it('has no Save for anything that is not editable text', () => {
        const { pagerEl } = setup({ tab: textTab, textSave: null });
        expect(buttons(pagerEl).map((b) => b.title)).toEqual(['Delete artefact', 'Ancestors', 'Descendants']);
        expect(pagerEl.children.find((c) => c.className === 'node-save-status')).toBeUndefined();
    });

    it('the remaining actions are plain buttons', () => {
        const { pagerEl } = setup({ tab: textTab, textSave: { onSave: vi.fn() } });
        expect(buttons(pagerEl).map((b) => b.className.includes('primary'))).toEqual([false, false, false]);
    });

    it('greys out Ancestors or Descendants for a node that has none', () => {
        const { pagerEl } = setup({ tab: textTab, lineage: { ancestors: false, descendants: true } });
        const [del, anc, desc] = buttons(pagerEl);
        expect([del.disabled, anc.disabled, desc.disabled]).toEqual([false, true, false]);
    });

    it('every button says its hover text to screen readers too', () => {
        const { pagerEl } = setup({ tab: textTab, textSave: { onSave: vi.fn() } });
        for (const b of buttons(pagerEl)) expect(b.attrs['aria-label']).toBe(b.title);
    });

    it('Delete deletes this tab\'s artefact, with its confirmation', () => {
        const { pagerEl, calls } = setup({ tab: textTab });
        buttons(pagerEl)[0].click();
        expect(calls.del).toEqual([['locality-1', '']]);
    });

    it('Ancestors and Descendants open for this tab\'s panel, beside the button pressed', () => {
        const { pagerEl, calls, controller } = setup({ tab: textTab });
        const [, anc, desc] = buttons(pagerEl);
        anc.click();
        desc.click();
        expect(calls.lineage).toEqual([[controller, 'ancestors', anc], [controller, 'descendants', desc]]);
    });

    it('keeps the autosave status line without a Save button', () => {
        const textSave = { onSave: vi.fn(), statusEl: null };
        const { pagerEl } = setup({ tab: textTab, textSave });
        expect(buttons(pagerEl).some((b) => b.title === 'Save changes')).toBe(false);
        expect(textSave.statusEl).toBe(pagerEl.children.find((c) => c.className === 'node-save-status'));
    });

    it('is hidden for a tab with nothing to show', () => {
        const { pagerEl } = setup({ tab: { kind: 'node', nodeId: null }, instances: [] });
        expect(pagerEl.hidden).toBe(true);
        expect(pagerEl.children).toEqual([]);
    });
});

describe('Ancestors and Descendants popovers', () => {
    const source = slice('function openLineageFor(', 'function closeLineagePopovers()');

    function setup() {
        const els = { ancestorsPopover: fakeElement('div'), descendantsPopover: fakeElement('div') };
        els.ancestorsPopover.style.display = 'none';
        els.descendantsPopover.style.display = 'none';
        const context = {
            focusedPanel: 'bottom',
            updateSharedUiForFocusedPanel: vi.fn(),
            ancestorsPopover: new IconPopover(), descendantsPopover: new IconPopover(),
            document: { getElementById: (id) => els[id] },
            window: { innerWidth: 1000 },
        };
        vm.createContext(context);
        vm.runInContext(source, context);
        return { els, context };
    }
    const anchor = (right, bottom) => ({ getBoundingClientRect: () => ({ right, bottom }) });

    it('focuses the tab\'s panel first, so the lists show that tab\'s node', () => {
        const { context } = setup();
        context.openLineageFor({ key: 'panel2' }, 'ancestors', anchor(500, 100));
        expect(context.focusedPanel).toBe('panel2');
        expect(context.updateSharedUiForFocusedPanel).toHaveBeenCalled();
    });

    it('opens just below the button, its right edge lined up with the button\'s', () => {
        const { context, els } = setup();
        context.openLineageFor({ key: 'bottom' }, 'ancestors', anchor(500, 100));
        const el = els.ancestorsPopover;
        expect(el.style.display).toBe('block');
        expect(el.style.position).toBe('fixed');
        expect(el.style.top).toBe('106px');
        expect(el.style.left).toBe('280px'); // 500 - the popover's 220px width
    });

    it('stays inside the window', () => {
        const { context, els } = setup();
        context.openLineageFor({ key: 'bottom' }, 'descendants', anchor(100, 40));
        expect(els.descendantsPopover.style.left).toBe('8px');
        context.openLineageFor({ key: 'bottom' }, 'descendants', anchor(2000, 40)); // closes
        context.openLineageFor({ key: 'bottom' }, 'descendants', anchor(2000, 40)); // opens again
        expect(els.descendantsPopover.style.left).toBe('772px'); // 1000 - 220 - 8
    });

    it('pressing the same button again closes it', () => {
        const { context, els } = setup();
        context.openLineageFor({ key: 'bottom' }, 'ancestors', anchor(500, 100));
        context.openLineageFor({ key: 'bottom' }, 'ancestors', anchor(500, 100));
        expect(els.ancestorsPopover.style.display).toBe('none');
    });

    it('pressing the same kind of button in ANOTHER panel moves the popover there instead of closing it', () => {
        const { context, els } = setup();
        context.openLineageFor({ key: 'panelA' }, 'ancestors', anchor(500, 100));
        context.openLineageFor({ key: 'panelB' }, 'ancestors', anchor(900, 300));
        const el = els.ancestorsPopover;
        expect(el.style.display).toBe('block');
        expect(el.style.top).toBe('306px');
        expect(el.style.left).toBe('680px'); // 900 - 220
        expect(context.focusedPanel).toBe('panelB');
    });

    it('and pressing it again in that panel then closes it', () => {
        const { context, els } = setup();
        context.openLineageFor({ key: 'panelA' }, 'ancestors', anchor(500, 100));
        context.openLineageFor({ key: 'panelB' }, 'ancestors', anchor(900, 300));
        context.openLineageFor({ key: 'panelB' }, 'ancestors', anchor(900, 300));
        expect(els.ancestorsPopover.style.display).toBe('none');
    });

    it('opening the other one closes the first', () => {
        const { context, els } = setup();
        context.openLineageFor({ key: 'bottom' }, 'ancestors', anchor(500, 100));
        context.openLineageFor({ key: 'bottom' }, 'descendants', anchor(560, 100));
        expect(els.ancestorsPopover.style.display).toBe('none');
        expect(els.descendantsPopover.style.display).toBe('block');
    });
});

describe('the text-editing tools', () => {
    const source = slice('PanelController.prototype.nodeToolsEl = function', 'PanelController.prototype.clear = ');

    function setup({ toolbar }) {
        const slot = fakeElement();
        const controller = {
            el: {
                headerType: { parentElement: { querySelector: (sel) => (sel === '.node-tools' ? slot : null) } },
                textDisplay: { querySelector: (sel) => (sel === '.editor-toolbar' ? toolbar : null) },
            },
        };
        const context = { PanelController: { prototype: {} } };
        vm.createContext(context);
        vm.runInContext(source, context);
        Object.assign(controller, context.PanelController.prototype);
        return { slot, controller };
    }

    it('moves the editor\'s toolbar into the left of the row under the tabs', () => {
        const toolbar = fakeElement('div');
        const { slot, controller } = setup({ toolbar });
        controller.moveEditorToolbar();
        expect(slot.children).toEqual([toolbar]);
    });

    it('leaves the slot empty for text with no tools (plain text)', () => {
        const { slot, controller } = setup({ toolbar: null });
        slot.children = [fakeElement()];
        controller.moveEditorToolbar();
        expect(slot.children).toEqual([]);
    });

    it('clears the tools when something else is shown', () => {
        const { slot, controller } = setup({ toolbar: fakeElement('div') });
        controller.moveEditorToolbar();
        controller.clearNodeTools();
        expect(slot.children).toEqual([]);
    });
});

describe('Save', () => {
    it('saves the editor\'s text for this artefact and reports on the status line', () => {
        const save = slice('this.textSave = (view ? view.editable : true)', 'mountEditor(view.text, isTei);');
        expect(save).toContain('onSave: () => this.textAutosave?.flush()');
        expect(save).toContain('window.pywebview.api.update_display_text(nodeId, value, resultsPath)');
        expect(save).toContain('flashStatus(saveState.statusEl, message, isError)');
    });
});
