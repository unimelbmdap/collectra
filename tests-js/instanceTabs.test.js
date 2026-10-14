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
