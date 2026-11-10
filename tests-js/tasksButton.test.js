const fs = require('node:fs');
const html = fs.readFileSync(require.resolve('../collectra/gui/index.html'), 'utf8');
const at = html.indexOf('id="tasksToolbarBtn"');
const button = html.slice(html.lastIndexOf('<button', at), html.indexOf('</button>', at));

describe('the Tasks toolbar button', () => {
    it('keeps opening the tasks popover and its labels', () => {
        expect(button).toContain('onclick="toggleTasksPopover()"');
        expect(button).toContain('title="Tasks"');
        expect(button).toContain('aria-label="Show tasks"');
    });

    it('no longer shows the plain list icon', () => {
        expect(button).not.toContain('M2 3.5h2M2 8h2M2 12.5h2');
        expect(button).not.toContain('M6 3.5h8M6 8h8M6 12.5h8');
    });

    it('shows a robot: a head, two eyes, an antenna and ears', () => {
        expect(button).toContain('data-icon="bot"');
        expect((button.match(/<rect /g) || []).length).toBe(1);
        expect((button.match(/<circle /g) || []).length).toBe(3);
        expect((button.match(/<path /g) || []).length).toBe(2);
    });

    it('is the same size and takes its colour from the toolbar like the other icons', () => {
        expect(button).toContain('width="16" height="16" viewBox="0 0 16 16"');
        expect(button).toContain('stroke="currentColor"');
    });
});
