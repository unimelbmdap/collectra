const fs = require('node:fs');
const html = fs.readFileSync(require.resolve('../collectra/gui/index.html'), 'utf8');
const rule = (selector) => {
    const start = html.indexOf(`        ${selector} {`);
    expect(start, selector).toBeGreaterThan(-1);
    return html.slice(start, html.indexOf('}', start));
};

describe('the vertical divider runs to the top of the window', () => {
    it('because nothing sits above the page content any more', () => {
        expect(html).not.toContain('titlebar-strip');
    });
});

describe('the horizontal dividers are hairlines like the vertical one', () => {
    it('is a 1px line, with a wider invisible area to grab', () => {
        const handle = rule('.image-panel-resize-handle');
        expect(handle).toContain('height: 1px;');
        expect(handle).toContain('background: var(--app-divider');
        const grab = rule('.image-panel-resize-handle::after');
        expect(grab).toContain('top: -4px;');
        expect(grab).toContain('height: 9px;');
        expect(grab).toContain('background: transparent;');
    });

    it('the vertical split handle is the same 1px line with the same grab area', () => {
        const handle = rule('.panel-resize-handle');
        expect(handle).toContain('width: 1px;');
        expect(handle).toContain('background: var(--app-divider');
        expect(handle).not.toContain('margin-right: -4px');
        const grab = rule('.panel-resize-handle::after');
        expect(grab).toContain('left: -4px;');
        expect(grab).toContain('width: 9px;');
    });

    it('only changes colour on hover and while dragging, never its thickness', () => {
        const start = html.indexOf('.image-panel-resize-handle:hover,');
        const hover = html.slice(start, html.indexOf('}', start));
        expect(hover).toContain('background: rgba(var(--app-primary-rgb');
        expect(hover).not.toContain('box-shadow');
        expect(hover).not.toMatch(/(width|height):/);
    });
});

describe('no second line right under the toolbar', () => {
    it('hides the top/bottom handle beside a collapsed panel, which has its own edge', () => {
        const start = html.indexOf('.collapsed + .image-panel-resize-handle');
        expect(start).toBeGreaterThan(-1);
        expect(html.slice(start, html.indexOf('}', start))).toContain('display: none;');
    });

    it('the handle really follows the top panel in the page', () => {
        const top = html.indexOf('id="imagePanelTop"');
        const handle = html.indexOf('id="imagePanelResizeHandle"');
        expect(handle).toBeGreaterThan(top);
        expect(html.slice(top, handle)).not.toMatch(/id="imagePanelBottom"/);
    });
});

describe('the sidebar divider is drawn inside the sidebar, like the others', () => {
    it('sits on the sidebar\'s own right edge so nothing is clipped', () => {
        const handle = rule('.sidebar-resize-handle');
        expect(handle).toContain('right: 0;');
        expect(handle).not.toContain('right: -4px;');
        const line = rule('.sidebar-resize-handle::after');
        expect(line).toContain('width: 1px;');
        expect(line).toContain('right: 0;');
        expect(line).toContain('height: 100%;');
        expect(line).not.toContain('translateX');
    });

    it('only changes colour on hover and while dragging', () => {
        const start = html.indexOf('.sidebar-resize-handle:hover::after');
        const hover = html.slice(start, html.indexOf('}', start));
        expect(hover).toContain('background: rgba(var(--app-primary-rgb');
        expect(hover).not.toMatch(/width:/);
    });
});
