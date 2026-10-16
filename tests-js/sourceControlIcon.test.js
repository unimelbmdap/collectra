const fs = require('node:fs');
const html = fs.readFileSync(require.resolve('../collectra/gui/index.html'), 'utf8');
const at = html.indexOf('id="gitToolbarBtn"');
const button = html.slice(html.lastIndexOf('<button', at), html.indexOf('</button>', at));

describe('Source control button uses the VS Code source-control icon', () => {
    it('keeps its id, title and label', () => {
        expect(button).toContain('id="gitToolbarBtn"');
        expect(button).toContain('title="History and Changes"');
        expect(button).toContain('aria-label="History and Changes"');
    });

    it('draws three circles joined by lines, not the git diamond', () => {
        expect((button.match(/<circle /g) || []).length).toBe(3);
        expect(button).not.toContain('rotate(45');
    });

    it('takes its colour from the toolbar like the other icons', () => {
        expect(button).toContain('stroke="currentColor"');
    });
});
