const { computeSplitLayout } = require('../collectra/gui/splitLayout.js');

describe('computeSplitLayout', () => {
    it('lays out left/right splits as a row', () => {
        expect(computeSplitLayout('left').flexDirection).toBe('row');
        expect(computeSplitLayout('right').flexDirection).toBe('row');
    });

    it('lays out above/below splits as a column', () => {
        expect(computeSplitLayout('above').flexDirection).toBe('column');
        expect(computeSplitLayout('below').flexDirection).toBe('column');
    });

    it('puts the new panel first for left and above', () => {
        expect(computeSplitLayout('left').newPanelFirst).toBe(true);
        expect(computeSplitLayout('above').newPanelFirst).toBe(true);
    });

    it('puts the new panel last for right and below', () => {
        expect(computeSplitLayout('right').newPanelFirst).toBe(false);
        expect(computeSplitLayout('below').newPanelFirst).toBe(false);
    });

    it('uses the row resize-handle class for left/right', () => {
        expect(computeSplitLayout('left').resizeHandleClass).toBe('panel-resize-handle');
        expect(computeSplitLayout('right').resizeHandleClass).toBe('panel-resize-handle');
    });

    it('uses the column resize-handle class for above/below', () => {
        expect(computeSplitLayout('above').resizeHandleClass).toBe('image-panel-resize-handle');
        expect(computeSplitLayout('below').resizeHandleClass).toBe('image-panel-resize-handle');
    });

    it('throws on an unknown direction', () => {
        expect(() => computeSplitLayout('diagonal')).toThrow();
    });
});
