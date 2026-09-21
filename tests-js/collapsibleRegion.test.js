const { CollapsibleRegion } = require('../collectra/gui/collapsibleRegion.js');

describe('CollapsibleRegion', () => {
    it('starts expanded by default', () => {
        const region = new CollapsibleRegion();
        expect(region.collapsed).toBe(false);
    });

    it('can start collapsed when constructed with collapsed: true', () => {
        const region = new CollapsibleRegion({ collapsed: true });
        expect(region.collapsed).toBe(true);
    });

    it('collapse() sets collapsed to true and returns it', () => {
        const region = new CollapsibleRegion();
        const result = region.collapse();
        expect(region.collapsed).toBe(true);
        expect(result).toBe(true);
    });

    it('collapse() is idempotent when already collapsed', () => {
        const region = new CollapsibleRegion({ collapsed: true });
        region.collapse();
        expect(region.collapsed).toBe(true);
    });

    it('expand() sets collapsed to false and returns it', () => {
        const region = new CollapsibleRegion({ collapsed: true });
        const result = region.expand();
        expect(region.collapsed).toBe(false);
        expect(result).toBe(false);
    });

    it('expand() is idempotent when already expanded', () => {
        const region = new CollapsibleRegion();
        region.expand();
        expect(region.collapsed).toBe(false);
    });

    it('toggle() flips collapsed from false to true', () => {
        const region = new CollapsibleRegion();
        const result = region.toggle();
        expect(region.collapsed).toBe(true);
        expect(result).toBe(true);
    });

    it('toggle() flips collapsed from true to false', () => {
        const region = new CollapsibleRegion({ collapsed: true });
        const result = region.toggle();
        expect(region.collapsed).toBe(false);
        expect(result).toBe(false);
    });

    it('toggle() called twice returns to the original state', () => {
        const region = new CollapsibleRegion();
        region.toggle();
        region.toggle();
        expect(region.collapsed).toBe(false);
    });
});