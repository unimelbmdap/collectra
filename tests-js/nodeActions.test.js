const fs = require('node:fs');
const vm = require('node:vm');
const { nodeActionButtons } = require('../collectra/gui/instancePager.js');
const html = fs.readFileSync(require.resolve('../collectra/gui/index.html'), 'utf8');
const between = (from, to) => html.slice(html.indexOf(from), html.indexOf(to, html.indexOf(from)));
const rule = (selector) => {
    const start = html.indexOf(`        ${selector} {`);
    expect(start, selector).toBeGreaterThan(-1);
    return html.slice(start, html.indexOf('}', start));
};

describe('nodeActionButtons', () => {
    const all = { hasAncestors: true, hasDescendants: true, canSave: true };

    it('is Delete, Ancestors, Descendants, in that order', () => {
        expect(nodeActionButtons(all).map((b) => b.id)).toEqual(['delete', 'ancestors', 'descendants']);
    });

    it('says what each one does when you hover it', () => {
        const titles = Object.fromEntries(nodeActionButtons(all).map((b) => [b.id, b.title]));
        expect(titles).toEqual({
            delete: 'Delete artefact',
            ancestors: 'Ancestors',
            descendants: 'Descendants',
        });
    });

    it('has no Save button for text fields', () => {
        expect(nodeActionButtons({ ...all, canSave: false }).map((b) => b.id)).toEqual(['delete', 'ancestors', 'descendants']);
    });

    it('greys out Ancestors or Descendants when the artefact has none', () => {
        const buttons = nodeActionButtons({ hasAncestors: false, hasDescendants: true, canSave: false });
        expect(buttons.find((b) => b.id === 'ancestors').disabled).toBe(true);
        expect(buttons.find((b) => b.id === 'descendants').disabled).toBe(false);
        expect(buttons.find((b) => b.id === 'delete').disabled).toBe(false);
    });

    it('has no primary save action', () => {
        expect(nodeActionButtons(all).filter((b) => b.primary)).toEqual([]);
    });
});

describe('one row under the tabs', () => {
    it('every header has a slot on the left for the editing tools, before the node details', () => {
        const headers = html.match(/<div class="node-content-header"[^>]*>[\s\S]*?<\/div>/g);
        expect(headers).toHaveLength(3);
        for (const header of headers) {
            expect(header).toContain('<span class="node-tools"></span>');
            expect(header.indexOf('node-tools')).toBeLessThan(header.indexOf('instance-pager'));
        }
    });

    it('the tools take the left, the node details and buttons sit at the right', () => {
        expect(rule('.node-content-header .node-tools')).toContain('flex: 1;');
        expect(rule('.node-content-header .instance-pager')).toContain('margin-left: auto;');
    });

    it('the separate Save row is gone', () => {
        expect(html).not.toContain('page-text-bar');
        expect(html).not.toContain('page-text-save');
    });

    it('the editor no longer carries its own Save row or status', () => {
        const mount = between('const mountEditor = (text, tei) => {', 'this.easyMDE = new EasyMDE({');
        expect(mount).not.toContain('Save');
        expect(mount).toContain('<textarea class="mde-textarea"></textarea>');
    });
});

describe('the action buttons', () => {
    it('are drawn from the shared definition, not hand-written text', () => {
        const render = between('renderPager() {', 'reconcileTabs(options)');
        expect(render).toContain('nodeActionButtons(');
        expect(render).toContain('lineageAvailability(tab.groupKey || tab.nodeId)');
        expect(render).not.toContain('`Delete ${state.name}`');
    });

    it('all look alike: 26px square icon buttons, with Save filled', () => {
        const btn = rule('.node-action-btn');
        expect(btn).toContain('width: 26px;');
        expect(btn).toContain('height: 26px;');
        expect(rule('.node-action-btn.primary')).toContain('background: var(--app-primary');
    });

    it('Ancestors and Descendants are gone from the top toolbar', () => {
        expect(html).not.toContain('id="ancestorsToolbarBtn"');
        expect(html).not.toContain('id="descendantsToolbarBtn"');
    });
});

describe('Ancestors and Descendants belong to each tab', () => {
    it('are opened from the tab\'s own row, next to the button, not from the top toolbar', () => {
        expect(html).toContain('function openLineageFor(');
        expect(html).not.toContain('function toggleAncestorsPopover(');
        expect(html).not.toContain('function toggleDescendantsPopover(');
        const render = between('renderPager() {', 'reconcileTabs(options)');
        expect(render).toContain("openLineageFor(this, 'ancestors', b)");
        expect(render).toContain("openLineageFor(this, 'descendants', b)");
    });

    it('the icons are defined once and used for the buttons', () => {
        const at = html.indexOf('const NODE_ACTION_ICONS');
        expect(at).toBeGreaterThan(-1);
        expect(at).toBeLessThan(html.indexOf('class PanelController'));
        const icons = between('const NODE_ACTION_ICONS', '\n        };');
        for (const id of ['delete', 'ancestors', 'descendants', 'save']) expect(icons).toContain(`${id}:`);
        expect(icons).toContain('fa-trash-can');
        expect(icons).toContain('fa-floppy-disk');
    });
});
