const fs = require('node:fs');
const { collapseToggleLabels, setCollapseToggleLabels } = require('../collectra/gui/collapseToggle.js');
const html = fs.readFileSync(require.resolve('../collectra/gui/index.html'), 'utf8');
const body = (name) => html.slice(html.indexOf(`function ${name}(`), html.indexOf('\n        }\n', html.indexOf(`function ${name}(`)));

describe('collapseToggleLabels', () => {
    it('says what pressing the button will do, not what it did', () => {
        expect(collapseToggleLabels(false, 'panel')).toEqual({title: 'Collapse panel', label: 'Collapse panel'});
        expect(collapseToggleLabels(true, 'panel')).toEqual({title: 'Expand panel', label: 'Expand panel'});
    });

    it('names whatever it collapses', () => {
        expect(collapseToggleLabels(false, 'sidebar').title).toBe('Collapse sidebar');
        expect(collapseToggleLabels(true, 'DAG').title).toBe('Expand DAG');
    });
});

describe('setCollapseToggleLabels', () => {
    const button = () => ({ attrs: {}, setAttribute(k, v) { this.attrs[k] = v; }, title: '' });

    it('sets both the hover text and the screen-reader label', () => {
        const b = button();
        setCollapseToggleLabels(b, true, 'panel');
        expect(b.title).toBe('Expand panel');
        expect(b.attrs['aria-label']).toBe('Expand panel');
        setCollapseToggleLabels(b, false, 'panel');
        expect(b.title).toBe('Collapse panel');
        expect(b.attrs['aria-label']).toBe('Collapse panel');
    });

    it('does nothing for a missing button', () => {
        expect(() => setCollapseToggleLabels(null, true, 'panel')).not.toThrow();
    });
});

describe('every collapse button keeps its hover text in step with its state', () => {
    it.each([
        ['applyPanelCollapse', "'panel'"],
    ])('%s', (fn, subject) => {
        expect(body(fn)).toContain(`setCollapseToggleLabels(`);
        expect(body(fn)).toContain(subject);
    });

    it('loads the helper', () => {
        expect(html).toContain('<script src="collapseToggle.js"></script>');
    });
});

describe('sidebarToggleTitle', () => {
    const { sidebarToggleTitle } = require('../collectra/gui/collapseToggle.js');

    it('is the VS Code wording with the shortcut for the platform', () => {
        expect(sidebarToggleTitle(true)).toBe('Toggle Primary Side Bar (⌘B)');
        expect(sidebarToggleTitle(false)).toBe('Toggle Primary Side Bar (Ctrl+B)');
    });
});

describe('isSidebarShortcut', () => {
    const { isSidebarShortcut } = require('../collectra/gui/collapseToggle.js');
    const key = (extra) => ({ key: 'b', metaKey: false, ctrlKey: false, shiftKey: false, altKey: false, ...extra });

    it('is Cmd+B or Ctrl+B, whatever the letter case', () => {
        expect(isSidebarShortcut(key({ metaKey: true }))).toBe(true);
        expect(isSidebarShortcut(key({ ctrlKey: true }))).toBe(true);
        expect(isSidebarShortcut(key({ metaKey: true, key: 'B' }))).toBe(true);
    });

    it('is not plain B, another key, or B with Shift/Alt added', () => {
        expect(isSidebarShortcut(key())).toBe(false);
        expect(isSidebarShortcut(key({ metaKey: true, key: 'f' }))).toBe(false);
        expect(isSidebarShortcut(key({ metaKey: true, shiftKey: true }))).toBe(false);
        expect(isSidebarShortcut(key({ ctrlKey: true, altKey: true }))).toBe(false);
    });

    it('copes with a key event that has no key', () => {
        expect(isSidebarShortcut({ metaKey: true })).toBe(false);
    });
});
