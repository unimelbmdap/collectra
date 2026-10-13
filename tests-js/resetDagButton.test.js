const fs = require('node:fs');
const html = fs.readFileSync(require.resolve('../collectra/gui/index.html'), 'utf8');
const start = html.indexOf('id="refitDagTabBtn"');
const button = html.slice(html.lastIndexOf('<button', start), html.indexOf('</button>', start));

it('keeps its id and still resets the DAG when clicked', () => {
    expect(button).toContain('onclick="resetDag()"');
});

it('is titled as a fit-to-view action', () => {
    expect(button).toContain('title="Fit DAG to view"');
    expect(button).toContain('aria-label="Fit DAG to view"');
});

it('shows the four-corner fit frame, not a circular refresh arrow', () => {
    expect(button).not.toContain('a5.5 5.5');
    const corners = button.match(/<path d="M[^"]*"/g) || [];
    expect(corners.length).toBe(4);
});

it('sits directly after the DAG button in the toolbar', () => {
    const ids = [...html.matchAll(/<button class="panel-toolbar-btn" id="(\w+)"/g)].map((m) => m[1]);
    expect(ids.indexOf('refitDagTabBtn')).toBe(ids.indexOf('dagToolbarBtn') + 1);
});
