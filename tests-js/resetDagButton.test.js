const fs = require('node:fs');
const html = fs.readFileSync(require.resolve('../collectra/gui/index.html'), 'utf8');

const containers = {
    'the top panel DAG': html.indexOf('<div id="dagContainer"'),
    'the bottom panel DAG tab': html.indexOf('<div id="dagTabContainer"'),
    'the DAG tab of a new panel': html.indexOf('<div class="dag-tab-container"'),
};

describe('Fit DAG to view lives on the DAG tab, not in the toolbar', () => {
    it('is gone from the toolbar', () => {
        expect(html).not.toContain('id="refitDagTabBtn"');
        expect(html).not.toContain("getElementById('refitDagTabBtn')");
    });

    it.each(Object.entries(containers))('is inside %s', (name, at) => {
        expect(at, name).toBeGreaterThan(-1);
        const tag = html.slice(at, at + 1200);
        const button = tag.slice(tag.indexOf('<button class="dag-fit-btn"'), tag.indexOf('</button>') + 9);
        expect(button).toContain('onclick="resetDag()"');
        expect(button).toContain('title="Fit DAG to view"');
        expect(button).toContain('aria-label="Fit DAG to view"');
    });

    it('draws the four-corner fit frame, not a circular refresh arrow', () => {
        const at = html.indexOf('<button class="dag-fit-btn"');
        const button = html.slice(at, html.indexOf('</button>', at));
        expect(button).not.toContain('a5.5 5.5');
        expect((button.match(/<path d="M/g) || []).length).toBe(4);
    });

    it('sits in the top-right corner of the DAG, above the graph', () => {
        const start = html.indexOf('        .dag-fit-btn {');
        expect(start).toBeGreaterThan(-1);
        const rule = html.slice(start, html.indexOf('}', start));
        expect(rule).toContain('position: absolute;');
        expect(rule).toContain('top: 8px;');
        expect(rule).toContain('right: 8px;');
        expect(rule).toMatch(/z-index:\s*\d+;/);
    });

    it('is only there while its DAG is on screen, because it lives inside the DAG container', () => {
        // Each button is a child of a container that is hidden when its DAG tab is not shown.
        for (const [name, at] of Object.entries(containers)) {
            expect(html.slice(at, at + 1200), name).toContain('dag-fit-btn');
        }
    });
});
