const { Tab, Panel } = require('../collectra/gui/panelSystem.js');
const { pagerState } = require('../collectra/gui/instancePager.js');
const { dagNodeOpenTarget } = require('../collectra/gui/dagNodeResolution.js');

const label = 'primary_label_unoriented';
const instances = ['primary_label_unoriented-b47ed', 'primary_label_unoriented1'];
const instancesFor = (map) => (groupKey) => map[groupKey] || [];

describe('Tab groupKey', () => {
    it('defaults to null and keeps an explicit DAG label', () => {
        expect(new Tab({ id: 't1', nodeId: 'a' }).groupKey).toBe(null);
        expect(new Tab({ id: 't1', nodeId: 'a1', groupKey: 'a' }).groupKey).toBe('a');
    });
});

describe('Panel.openOrActivateGroup', () => {
    it('opens one tab per DAG label with the chosen instance selected', () => {
        const panel = new Panel();
        const tab = panel.openOrActivateGroup({ id: 't1', groupKey: label, nodeId: instances[0], label: 'Primary Label Unoriented' });
        expect(panel.tabs).toEqual([tab]);
        expect(tab.groupKey).toBe(label);
        expect(tab.nodeId).toBe(instances[0]);
        expect(tab.label).toBe('Primary Label Unoriented');
        expect(panel.activeTabId).toBe('t1');
    });

    it('re-opening the same label reuses its tab and never duplicates it', () => {
        const panel = new Panel();
        panel.openOrActivateGroup({ id: 't1', groupKey: label, nodeId: instances[0] });
        panel.openOrActivateGroup({ id: 't2', groupKey: 'other', nodeId: 'other1' });
        const again = panel.openOrActivateGroup({ id: 't3', groupKey: label, nodeId: instances[1] });
        expect(panel.tabs.length).toBe(2);
        expect(again.id).toBe('t1');
        expect(panel.activeTabId).toBe('t1');
        expect(again.nodeId).toBe(instances[1]);
    });

    it('keeps the selected instance when re-opened without naming one', () => {
        const panel = new Panel();
        panel.openOrActivateGroup({ id: 't1', groupKey: label, nodeId: instances[1] });
        const again = panel.openOrActivateGroup({ id: 't2', groupKey: label });
        expect(again.nodeId).toBe(instances[1]);
    });

    it('does not repurpose the active tab of a different label', () => {
        const panel = new Panel();
        panel.openOrActivateGroup({ id: 't1', groupKey: 'a', nodeId: 'a1' });
        panel.openOrActivateGroup({ id: 't2', groupKey: 'b', nodeId: 'b1' });
        expect(panel.tabs.map((t) => t.groupKey)).toEqual(['a', 'b']);
        expect(panel.getTab('t1').nodeId).toBe('a1');
    });
});

describe('Panel.selectInstance', () => {
    it('changes the instance a label tab shows, leaving its label and position alone', () => {
        const panel = new Panel();
        panel.openOrActivateGroup({ id: 't1', groupKey: label, nodeId: instances[0] });
        panel.selectInstance('t1', instances[1]);
        expect(panel.getTab('t1').nodeId).toBe(instances[1]);
        expect(panel.getTab('t1').groupKey).toBe(label);
        expect(panel.tabs.length).toBe(1);
    });

    it('ignores an unknown tab', () => {
        const panel = new Panel();
        expect(() => panel.selectInstance('nope', 'x')).not.toThrow();
    });
});

describe('Panel.reconcileGroupTabs', () => {
    const build = () => {
        const panel = new Panel();
        panel.openOrActivateGroup({ id: 't1', groupKey: label, nodeId: instances[1] });
        panel.addTab(new Tab({ id: 'dag', nodeId: null, label: 'DAG', kind: 'dag' }));
        return panel;
    };

    it('leaves a tab alone while its selected instance still exists', () => {
        const panel = build();
        const result = panel.reconcileGroupTabs(instancesFor({ [label]: instances }));
        expect(panel.getTab('t1').nodeId).toBe(instances[1]);
        expect(result).toEqual({ closed: [], changed: [] });
    });

    it('moves to the neighbouring instance when the selected one is gone', () => {
        const panel = build();
        panel.reconcileGroupTabs(instancesFor({ [label]: instances })); // remembers position 1
        const result = panel.reconcileGroupTabs(instancesFor({ [label]: [instances[0]] }));
        expect(panel.getTab('t1').nodeId).toBe(instances[0]);
        expect(result.changed).toEqual(['t1']);
    });

    it('keeps the tab, blank, when its label has no instances left', () => {
        const panel = build();
        const result = panel.reconcileGroupTabs(instancesFor({}));
        expect(panel.getTab('t1')).not.toBe(null);
        expect(panel.getTab('t1').nodeId).toBe(null);
        expect(panel.getTab('t1').groupKey).toBe(label);
        expect(result).toEqual({ closed: [], changed: ['t1'] });
        expect(panel.getTab('dag')).not.toBe(null);
    });

    it('fills a blank tab again as soon as the label has an instance', () => {
        const panel = build();
        panel.reconcileGroupTabs(instancesFor({}));
        const result = panel.reconcileGroupTabs(instancesFor({ [label]: ['p2-a', 'p2-b'] }));
        expect(panel.getTab('t1').nodeId).toBe('p2-a');
        expect(result.changed).toEqual(['t1']);
    });

    it('does not report an already blank tab as changed again', () => {
        const panel = build();
        panel.reconcileGroupTabs(instancesFor({}));
        expect(panel.reconcileGroupTabs(instancesFor({}))).toEqual({ closed: [], changed: [] });
    });

    it('never touches DAG tabs', () => {
        const panel = build();
        panel.reconcileGroupTabs(instancesFor({}));
        expect(panel.getTab('dag').kind).toBe('dag');
    });

    it('selects the first instance on a page switch when asked to', () => {
        const panel = build();
        panel.reconcileGroupTabs(instancesFor({ [label]: ['p2-a', 'p2-b'] }), { preferFirst: true });
        expect(panel.getTab('t1').nodeId).toBe('p2-a');
    });
});

describe('pagerState', () => {
    it('shows just the name when there is nothing to step between', () => {
        expect(pagerState(['only'], 'only')).toEqual({
            name: 'only', count: 1, position: 1, showControls: false, prevId: null, nextId: null, text: '',
        });
    });

    it('steps between instances in the given order, stopping at the ends', () => {
        const ids = ['a', 'b', 'c'];
        expect(pagerState(ids, 'a')).toMatchObject({ position: 1, prevId: null, nextId: 'b', showControls: true, text: '1 / 3' });
        expect(pagerState(ids, 'b')).toMatchObject({ position: 2, prevId: 'a', nextId: 'c' });
        expect(pagerState(ids, 'c')).toMatchObject({ position: 3, prevId: 'b', nextId: null, text: '3 / 3' });
    });

    it('shows no controls for an id that is not among the instances', () => {
        expect(pagerState(['a', 'b'], 'zzz')).toMatchObject({ name: 'zzz', showControls: false, prevId: null, nextId: null });
    });

    it('handles an empty list', () => {
        expect(pagerState([], 'x')).toMatchObject({ count: 0, showControls: false });
    });
});

describe('dagNodeOpenTarget', () => {
    const pageNodeIds = new Map([
        [label, instances],
        ['specimen_sheet', ['specimen_sheet']],
        ['field_detector_output', []],
    ]);

    it('opens the label, on its first instance, when the page has instances', () => {
        expect(dagNodeOpenTarget(label, pageNodeIds, true)).toEqual({ groupKey: label, nodeId: instances[0] });
    });

    it('is not clickable when the page has no instance of that label', () => {
        expect(dagNodeOpenTarget('field_detector_output', pageNodeIds, true)).toBe(null);
        expect(dagNodeOpenTarget('never_seen', pageNodeIds, true)).toBe(null);
    });

    it('still opens by label before any page is bound', () => {
        expect(dagNodeOpenTarget('specimen_sheet', new Map(), false)).toEqual({ groupKey: 'specimen_sheet', nodeId: 'specimen_sheet' });
    });
});

describe('blankTabMessage', () => {
    const { blankTabMessage } = require('../collectra/gui/instancePager.js');

    it('says which node the page has none of, using the tab label', () => {
        expect(blankTabMessage({ label: 'Primary Label Unoriented', groupKey: label }))
            .toBe('This page has no Primary Label Unoriented.');
    });

    it('falls back to the DAG label when the tab has no display label', () => {
        expect(blankTabMessage({ label: null, groupKey: label })).toBe(`This page has no ${label}.`);
    });
});

describe('blank tabs in the page', () => {
    const html = require('node:fs').readFileSync(require.resolve('../collectra/gui/index.html'), 'utf8');

    it('show a message instead of asking the backend for a missing node', () => {
        expect(html).toContain("if (tab.kind === 'node' && !tab.nodeId) {");
        expect(html).toContain('blankTabMessage(tab)');
    });

    it('hide the pager for a blank tab', () => {
        expect(html).toContain("if (!tab || tab.kind !== 'node' || !tab.nodeId) {");
    });
});

describe('tabTypeTitle', () => {
    const { tabTypeTitle } = require('../collectra/gui/instancePager.js');
    const types = new Map([['age', 'collectra.Text'], ['specimen_sheet', 'collectra.Image']]);
    const same = (id) => id;

    it('is the data type of the tab\'s node, for the tooltip', () => {
        expect(tabTypeTitle({ kind: 'node', groupKey: 'age', nodeId: 'age1' }, types, same)).toBe('collectra.Text');
    });

    it('finds the type through the shown instance when the tab has no label of its own', () => {
        const toLabel = (id) => (id === 'age1' ? 'age' : id);
        expect(tabTypeTitle({ kind: 'node', groupKey: null, nodeId: 'age1' }, types, toLabel)).toBe('collectra.Text');
    });

    it('is empty for a DAG tab, a blank tab and an unknown node', () => {
        expect(tabTypeTitle({ kind: 'dag', groupKey: null, nodeId: null }, types, same)).toBe('');
        expect(tabTypeTitle({ kind: 'node', groupKey: 'zzz', nodeId: null }, types, same)).toBe('');
        expect(tabTypeTitle({ kind: 'node', groupKey: 'zzz', nodeId: 'zzz1' }, types, same)).toBe('');
    });

    it('still names the type for a blank tab whose label the page knows', () => {
        expect(tabTypeTitle({ kind: 'node', groupKey: 'age', nodeId: null }, types, same)).toBe('collectra.Text');
    });
});

describe('the data type no longer sits under the tabs', () => {
    const html = require('node:fs').readFileSync(require.resolve('../collectra/gui/index.html'), 'utf8');
    const strip = html.slice(html.indexOf('function renderTabStrip('), html.indexOf('function activateTabInPanel'));

    it('hands the type to the same styled hover as the DAG, not a plain browser tooltip', () => {
        expect(strip).toContain('tabTypeTitle(tab, pageNodeTypes, resolveDagLabel)');
        expect(strip).toContain('data-type-title="${escapeHtml(typeTitle)}"');
        expect(strip).toContain('onmouseenter="showTabTooltip(this)"');
        expect(strip).toContain('onmouseleave="hideTabTooltip()"');
        expect(strip).not.toMatch(/class="tab[^"]*"[^>]*\stitle=/s);
    });

    it('does not let the label\'s own tooltip get in the way', () => {
        expect(strip).not.toContain('<span title="${escapeHtml(tab.nodeId)}">');
    });

    it('hides the type chip under the tabs', () => {
        const start = html.indexOf('        .node-content-header .node-content-header-type {');
        expect(html.slice(start, html.indexOf('}', start))).toContain('display: none;');
    });

    it('keeps the row for the pager, and hides it when there is no pager', () => {
        expect(html).toContain('.node-content-header:has(.instance-pager[hidden]) {');
        expect(html).not.toContain('.node-content-header:has(.node-content-header-type:empty) {');
    });

    it('has one tooltip card that looks like the DAG one', () => {
        const at = html.indexOf('id="tabTooltip"');
        expect(at).toBeGreaterThan(-1);
        const tag = html.slice(html.lastIndexOf('<div', at), html.indexOf('>', at));
        expect(tag).toContain('class="card"');
        expect(tag).toContain('var(--app-secondary');
        expect(tag).toContain('position:fixed');
    });
});

describe('the tab hover behaves', () => {
    const vm = require('node:vm');
    const html = require('node:fs').readFileSync(require.resolve('../collectra/gui/index.html'), 'utf8');
    const start = html.indexOf('// ── TAB TOOLTIP');
    const source = html.slice(start, html.indexOf('// ── END TAB TOOLTIP', start));
    const { typeChipHtml } = require('../collectra/gui/deleteDialog.js');

    function setup() {
        const tooltip = { style: {}, innerHTML: '' };
        const timer = require('../collectra/gui/hoverTimer.js');
        const context = { typeChipHtml, ...timer, document: { getElementById: () => tooltip } };
        vm.createContext(context);
        vm.runInContext(source, context);
        return { tooltip, context };
    }
    const tab = (type) => ({
        dataset: { typeTitle: type },
        getBoundingClientRect: () => ({ left: 120, bottom: 60 }),
    });

    it('shows the type chip just below the tab', () => {
        const { tooltip, context } = setup();
        context.showTabTooltip(tab('collectra.Text'));
        expect(tooltip.style.display).toBe('block');
        expect(tooltip.innerHTML).toContain('>Text</span>');
        expect(tooltip.style.left).toBe('120px');
        expect(tooltip.style.top).toBe('66px');
    });

    it('shows nothing for a tab with no type, like the DAG tab', () => {
        const { tooltip, context } = setup();
        context.showTabTooltip(tab(''));
        expect(tooltip.style.display).toBe('none');
    });

    it('hides when the pointer leaves', () => {
        const { tooltip, context } = setup();
        context.showTabTooltip(tab('collectra.Text'));
        context.hideTabTooltip();
        expect(tooltip.style.display).toBe('none');
    });
});

describe('the DAG tab shows no node details', () => {
    const html = require('node:fs').readFileSync(require.resolve('../collectra/gui/index.html'), 'utf8');
    const show = html.slice(html.indexOf('async showActiveTabContent() {'), html.indexOf('renderDagTab() {'));
    const dagBranch = show.slice(show.indexOf("if (tab.kind === 'dag') {"), show.indexOf("if (tab.kind === 'node' && !tab.nodeId)"));

    it('clears whatever the last node left in the header, including its type', () => {
        expect(dagBranch).toContain('this.clearHeader();');
    });

    it('also clears the selected node, so nothing from it is highlighted or shown', () => {
        expect(dagBranch).toContain('this.selectedNodeId = null;');
    });
});

describe('the hidden type label under the tabs', () => {
    const html = require('node:fs').readFileSync(require.resolve('../collectra/gui/index.html'), 'utf8');

    it('is no longer filled in anywhere, since the type shows on hover instead', () => {
        expect(html).not.toMatch(/headerType\.textContent/);
    });

    it('stays in the page as the anchor the row\'s slots are found from', () => {
        expect(html).toContain('this.el.headerType.parentElement');
    });
});

describe('Panel.removeNodeTabsNotIn', () => {
    const build = () => {
        const panel = new Panel();
        panel.addTab(new Tab({ id: 'dag', nodeId: null, label: 'DAG', kind: 'dag' }));
        panel.openOrActivateGroup({ id: 'a', groupKey: 'locality', nodeId: 'locality-1', label: 'Locality' });
        panel.openOrActivateGroup({ id: 'b', groupKey: 'locality-18925dcf-6676-4990-8e88-6a5c0a7db284', nodeId: null, label: 'Locality-18925dcf' });
        panel.openOrActivateGroup({ id: 'c', groupKey: 'genus', nodeId: null, label: 'Genus' });
        return panel;
    };
    const labels = new Set(['locality', 'genus']);

    it('drops node tabs whose label is not a pipeline label, such as one named after an instance id', () => {
        const panel = build();
        const removed = panel.removeNodeTabsNotIn(labels);
        expect(removed).toBe(1);
        expect(panel.tabs.map((t) => t.groupKey || t.kind)).toEqual(['dag', 'locality', 'genus']);
    });

    it('never touches DAG tabs, and keeps the active tab when it survives', () => {
        const panel = build();
        panel.activateTab('a');
        panel.removeNodeTabsNotIn(labels);
        expect(panel.getTab('dag')).not.toBe(null);
        expect(panel.activeTabId).toBe('a');
    });

    it('moves the active tab to one that remains when the active one is dropped', () => {
        const panel = build();
        panel.activateTab('b');
        panel.removeNodeTabsNotIn(labels);
        expect(panel.getTab(panel.activeTabId)).not.toBe(null);
    });

    it('does nothing when it has no labels to compare with (the pipeline is not loaded yet)', () => {
        const panel = build();
        expect(panel.removeNodeTabsNotIn(new Set())).toBe(0);
        expect(panel.tabs).toHaveLength(4);
    });

    it('does nothing when every tab is fine', () => {
        const panel = new Panel();
        panel.openOrActivateGroup({ id: 'a', groupKey: 'locality', nodeId: null, label: 'Locality' });
        expect(panel.removeNodeTabsNotIn(labels)).toBe(0);
    });
});
