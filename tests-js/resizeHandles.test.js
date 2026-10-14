// Runs the page's real drag code against stand-in elements.
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync(require.resolve('../collectra/gui/index.html'), 'utf8');
const start = html.indexOf('function wireGenericResizeHandle(');
const source = html.slice(start, html.indexOf('// Splits the focused panel', start));

function setup({ direction, first, wrapperSize }) {
    const listeners = {};
    const classes = new Set();
    const wrapper = {
        classList: { contains: (c) => c === `split-${direction}` },
        getBoundingClientRect: () => wrapperSize,
    };
    const handle = {
        parentElement: wrapper,
        previousElementSibling: first,
        classList: { add: (c) => classes.add(c), remove: (c) => classes.delete(c) },
        addEventListener: (name, fn) => { listeners[name] = fn; },
    };
    const docListeners = {};
    const context = {
        PANEL_MIN_SIZE: 120,
        scheduleArrangementSave: vi.fn(),
        document: {
            body: { style: {} },
            addEventListener: (n, f) => { docListeners[n] = f; },
            removeEventListener: (n) => { delete docListeners[n]; },
        },
    };
    vm.createContext(context);
    vm.runInContext(source, context);
    return { handle, listeners, docListeners, context, classes };
}

const box = (width, height) => ({ getBoundingClientRect: () => ({ width, height }), style: {} });
const drag = (t, from, to, axis) => {
    t.listeners.mousedown({ clientX: from.x, clientY: from.y, preventDefault() {} });
    t.docListeners.mousemove({ clientX: to.x, clientY: to.y });
};

describe('a divider resizes what is directly before it, in its own direction', () => {
    it('in a column, dragging up and down changes the height of the thing above it', () => {
        const above = box(900, 300);
        const t = setup({ direction: 'column', first: above, wrapperSize: { width: 900, height: 800 } });
        t.context.wireGenericResizeHandle(t.handle);

        drag(t, { x: 100, y: 300 }, { x: 100, y: 400 });

        expect(above.style.flex).toBe('0 1 50.00%'); // 300 + 100 of vertical movement = 400 of 800
    });

    it('in a row, dragging left and right changes the width', () => {
        const left = box(300, 500);
        const t = setup({ direction: 'row', first: left, wrapperSize: { width: 1000, height: 500 } });
        t.context.wireGenericResizeHandle(t.handle);

        drag(t, { x: 300, y: 50 }, { x: 380, y: 999 });

        expect(left.style.flex).toBe('0 1 38.00%'); // 380 of 1000; vertical movement is ignored
    });

    it('is not fooled by the panel and direction it was created with, once the layout has changed', () => {
        // Created as a row divider next to an old panel; later that panel was wrapped
        // and the divider became the one between a row of panels and the panel below.
        const stalePanel = box(300, 400);
        const rowOfPanelsAbove = box(900, 300);
        const t = setup({ direction: 'column', first: rowOfPanelsAbove, wrapperSize: { width: 900, height: 800 } });
        t.context.wireGenericResizeHandle(t.handle, 'row', stalePanel);

        drag(t, { x: 100, y: 300 }, { x: 100, y: 360 });

        expect(rowOfPanelsAbove.style.flex).toBe('0 1 45.00%');
        expect(stalePanel.style.flex).toBeUndefined();
    });

    it('keeps both sides at least the minimum size', () => {
        const above = box(900, 300);
        const t = setup({ direction: 'column', first: above, wrapperSize: { width: 900, height: 800 } });
        t.context.wireGenericResizeHandle(t.handle);

        drag(t, { x: 0, y: 300 }, { x: 0, y: 5000 });
        expect(above.style.flex).toBe('0 1 85.00%'); // 680 of 800 (leaves 120)

        drag(t, { x: 0, y: 300 }, { x: 0, y: -5000 });
        expect(above.style.flex).toBe('0 1 15.00%'); // 120 of 800
    });

    it('saves the layout when the drag ends', () => {
        const t = setup({ direction: 'column', first: box(900, 300), wrapperSize: { width: 900, height: 800 } });
        t.context.wireGenericResizeHandle(t.handle);
        t.listeners.mousedown({ clientX: 0, clientY: 0, preventDefault() {} });
        t.docListeners.mouseup();
        expect(t.context.scheduleArrangementSave).toHaveBeenCalled();
    });
});

describe('in the page', () => {
    it('creates dividers without remembering a panel or direction', () => {
        expect(html).not.toContain('wireGenericResizeHandle(resizeHandle, layout.flexDirection');
        expect(html.match(/wireGenericResizeHandle\(resizeHandle\);/g)).toHaveLength(2);
    });

    it('the main top/bottom divider also resizes whatever is directly above it', () => {
        const main = html.slice(html.indexOf("const handle = document.getElementById('imagePanelResizeHandle');"), html.indexOf('(function() {', html.indexOf("const handle = document.getElementById('imagePanelResizeHandle');")));
        expect(main).toContain('handle.previousElementSibling');
        expect(main).not.toContain("document.getElementById('imagePanelTop')");
    });
});

describe('sizes are proportions, so they follow the window like VS Code', () => {
    it('a dragged panel keeps its share when the window changes size', () => {
        const above = box(900, 300);
        const t = setup({ direction: 'column', first: above, wrapperSize: { width: 900, height: 800 } });
        t.context.wireGenericResizeHandle(t.handle);
        drag(t, { x: 0, y: 300 }, { x: 0, y: 500 });
        // A percentage of the container, not a fixed pixel size, and it may shrink.
        expect(above.style.flex).toMatch(/^0 1 [\d.]+%$/);
        expect(above.style.flex).not.toContain('px');
    });
});
