const fs = require('node:fs');
const html = fs.readFileSync(require.resolve('../collectra/gui/index.html'), 'utf8');
const rule = (selector) => {
    const start = html.indexOf(`        ${selector} {`);
    expect(start, selector).toBeGreaterThan(-1);
    return html.slice(start, html.indexOf('}', start));
};

describe('no padding or frames around the work area', () => {
    it('the content area runs edge to edge', () => {
        const body = rule('.node-content-body');
        expect(body).toContain('padding: 0;');
        expect(body).toContain('gap: 0;');
    });

    it('text and node views have no border or rounded corners of their own', () => {
        const view = rule('.node-content-display,\n        .page-text-display');
        expect(view).toContain('border: none;');
        expect(view).toContain('border-radius: 0;');
    });

    it('the image view has no rounded corners either', () => {
        expect(rule('.page-artefact-display')).toContain('border-radius: 0;');
    });

    it('the editor adds no second frame or inner padding', () => {
        const editor = rule('.page-text-display .EasyMDEContainer .CodeMirror,\n        .node-content-display .EasyMDEContainer .CodeMirror');
        expect(editor).toContain('border: none;');
        expect(editor).toContain('border-radius: 0;');
        expect(editor).toContain('padding: 0;');
    });

    it('the text keeps only a small inset of its own', () => {
        expect(rule('.page-text-display .CodeMirror-lines,\n        .node-content-display .CodeMirror-lines')).toContain('padding: 8px 12px;');
    });

    it('the formatting tools carry none of EasyMDE\'s own box, now they sit in the row under the tabs', () => {
        const bar = rule('.node-tools .editor-toolbar');
        expect(bar).toContain('border: none;');
        expect(bar).toContain('padding: 0;');
        expect(bar).toContain('background: transparent;');
    });
});
