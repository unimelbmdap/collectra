const fs = require('node:fs');
const html = fs.readFileSync(require.resolve('../collectra/gui/index.html'), 'utf8');
const at = html.indexOf('id="newPanelToolbarBtn"');
const button = html.slice(html.lastIndexOf('<button', at), html.indexOf('</button>', at));

describe('the + button is now Customize Layout', () => {
    it('keeps opening the panel arrangement popover', () => {
        expect(button).toContain('onclick="showSplitViewPopover(event)"');
        expect(button).toContain('id="newPanelToolbarBtn"');
    });

    it('says Customize Layout on hover and to screen readers', () => {
        expect(button).toContain('title="Customize Layout"');
        expect(button).toContain('aria-label="Customize Layout"');
        expect(button).not.toContain('New Panel');
    });

    it('shows a window-layout icon, not a plus', () => {
        expect(button).not.toContain('M8 2.5v11M2.5 8h11');
        expect(button).toContain('<rect');
        expect((button.match(/<path /g) || []).length).toBe(2);
    });

    it('takes its colour from the toolbar like the other icons', () => {
        expect(button).toContain('stroke="currentColor"');
    });
});
