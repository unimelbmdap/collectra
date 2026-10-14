const fs = require('node:fs');
const html = fs.readFileSync(require.resolve('../collectra/gui/index.html'), 'utf8');
const start = html.indexOf('        .app-container {');
const rule = html.slice(start, html.indexOf('}', start));

describe('the app fills the window with no frame around it', () => {
    it('has no margin, so it reaches the window edges', () => {
        expect(rule).toContain('margin: 0;');
    });

    it('has no outer border', () => {
        expect(rule).not.toMatch(/border:\s*1px/);
    });

    it('has no rounded corners', () => {
        expect(rule).not.toContain('border-radius: 4px');
    });
});
