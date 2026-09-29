// IconPopover — pure state for the show/hide toggle behind the icon
// toolbar's floating popovers (Ancestors, Descendants). Same DOM-free/
// DOM-adapter split as CollapsibleRegion (collapsibleRegion.js): this
// class holds no document/window access so it's unit-testable, and a
// thin adapter in index.html applies `open` to the actual popover
// element's display style and its trigger button's active class.
// Deliberately a separate class from CollapsibleRegion rather than reused
// — CollapsibleRegion's DOM adapter is built for docked, resizable
// columns (sidebar, DAG, right panel); a floating popover that stacks
// with a sibling and dismisses on outside-click needs different adapter
// logic, even though the state shape (open vs collapsed) is the same
// two-state toggle.
class IconPopover {
    constructor({ open = false } = {}) {
        this.open = open;
    }

    show() {
        this.open = true;
        return this.open;
    }

    hide() {
        this.open = false;
        return this.open;
    }

    toggle() {
        this.open = !this.open;
        return this.open;
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { IconPopover };
}