// CollapsibleRegion — pure state for the show/hide-to-a-thin-strip toggle
// pattern used across the app's edge regions (sidebar, DAG, Artefacts/Tasks
// panel, and eventually the lineage columns). DOM-free so it's unit
// testable without a browser; a thin DOM adapter applies `collapsed` to
// the actual element's classes/icon/inline size. Loaded as a plain classic
// script in index.html (defines a global, same as everything else there),
// and via require() in tests/.
class CollapsibleRegion {
    constructor({ collapsed = false } = {}) {
        this.collapsed = collapsed;
    }

    collapse() {
        this.collapsed = true;
        return this.collapsed;
    }

    expand() {
        this.collapsed = false;
        return this.collapsed;
    }

    toggle() {
        this.collapsed = !this.collapsed;
        return this.collapsed;
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { CollapsibleRegion };
}