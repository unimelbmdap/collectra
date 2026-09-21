const { IconPopover } = require('../collectra/gui/iconPopover.js');

describe('IconPopover', () => {
    it('starts closed by default', () => {
        const popover = new IconPopover();
        expect(popover.open).toBe(false);
    });

    it('can start open when constructed with open: true', () => {
        const popover = new IconPopover({ open: true });
        expect(popover.open).toBe(true);
    });

    it('show() sets open to true and returns it', () => {
        const popover = new IconPopover();
        const result = popover.show();
        expect(popover.open).toBe(true);
        expect(result).toBe(true);
    });

    it('show() is idempotent when already open', () => {
        const popover = new IconPopover({ open: true });
        popover.show();
        expect(popover.open).toBe(true);
    });

    it('hide() sets open to false and returns it', () => {
        const popover = new IconPopover({ open: true });
        const result = popover.hide();
        expect(popover.open).toBe(false);
        expect(result).toBe(false);
    });

    it('hide() is idempotent when already closed', () => {
        const popover = new IconPopover();
        popover.hide();
        expect(popover.open).toBe(false);
    });

    it('toggle() flips open from false to true', () => {
        const popover = new IconPopover();
        const result = popover.toggle();
        expect(popover.open).toBe(true);
        expect(result).toBe(true);
    });

    it('toggle() flips open from true to false', () => {
        const popover = new IconPopover({ open: true });
        const result = popover.toggle();
        expect(popover.open).toBe(false);
        expect(result).toBe(false);
    });

    it('toggle() called twice returns to the original state', () => {
        const popover = new IconPopover();
        popover.toggle();
        popover.toggle();
        expect(popover.open).toBe(false);
    });
});