const fs = require('node:fs');
const html = fs.readFileSync(require.resolve('../collectra/gui/index.html'), 'utf8');
const block = (selector) => {
    const start = html.indexOf(`selector: '${selector}'`);
    expect(start, selector).toBeGreaterThan(-1);
    return html.slice(start, html.indexOf('\n                    },', start));
};

describe('DAG node colours', () => {
    it('are filled at 35% strength, the same as a selected box on the image', () => {
        expect(block('node')).toContain("'background-opacity': 0.35,");
    });

    it('keep the label colour per node', () => {
        expect(block('node')).toContain("'background-color': 'data(nodeColor)',");
    });

    it('leave the label text fully opaque and unchanged', () => {
        const node = block('node');
        expect(node).toContain("'color': '#22313b',");
        expect(node).not.toMatch(/'text-opacity'/);
    });

    it('leave the rings on open and active nodes as they were', () => {
        expect(block('node.cy-open')).toContain("'border-width': 2, 'border-color': colors.orange");
        expect(block('node.cy-active')).toContain("'border-width': 3, 'border-color': colors.orange");
    });

    it('fill nodes that are not on this page at 10%, with their label as faint as before', () => {
        const dimmed = block('node.dimmed');
        expect(dimmed).toContain("'background-opacity': 0.10,");
        expect(dimmed).toContain("'text-opacity': 0.38,");
        expect(dimmed).toContain("'events': 'no',"); // still not clickable
    });

    it('no longer fade the whole node, which would stack on top of the fill strength', () => {
        expect(block('node.dimmed')).not.toContain("'opacity': 0.38");
    });

    it('leave the edges of inactive nodes faded as before', () => {
        expect(block('edge.dimmed')).toContain("'opacity': 0.15");
    });
});
