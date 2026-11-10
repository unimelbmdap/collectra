const { Tab, Panel } = require('../collectra/gui/panelSystem.js');
const { serializeArrangement, restorePanelTabs } = require('../collectra/gui/arrangement.js');
const {
    tableColumnDefs, tableRowIndex, createTableView, cellEditorFor, parseClipboard, formatClipboard, rangeBounds, pasteTargets,
} = require('../collectra/gui/tableView.js');

globalThis.window = { addEventListener() {}, removeEventListener() {} };
const gridEl = () => ({ addEventListener() {}, removeEventListener() {} });

describe('tableColumnDefs', () => {
    it('pins the file name and makes one text column per artefact label', () => {
        const defs = tableColumnDefs(['title', 'body.text'], (label) => label.toUpperCase());

        expect(defs.map((d) => d.field)).toEqual(['name', 'title', 'body.text']);
        expect(defs[0]).toMatchObject({ headerName: 'File', pinned: 'left' });
        expect(defs[2]).toMatchObject({ headerName: 'BODY.TEXT', tooltipField: 'body.text' });
    });
});

describe('tableRowIndex', () => {
    it('is the index when that page has a row, else null', () => {
        const rows = [{ index: 0 }, { index: 2 }];
        expect(tableRowIndex(rows, 2)).toBe(2);
        expect(tableRowIndex(rows, 1)).toBeNull();
        expect(tableRowIndex(rows, -1)).toBeNull();
    });
});

// A stand-in for AG Grid that keeps rows and their selection. Like the real
// grid it reports a selection later, when flush() is called.
function fakeGridLib() {
    const lib = { options: null, nodes: [], pending: [], flush: () => lib.pending.splice(0).forEach((fn) => fn()) };
    lib.createGrid = (el, options) => {
        lib.options = options;
        const api = {
            setGridOption: (name, value) => {
                options[name] = value;
                if (name === 'rowData') {
                    lib.nodes = value.map((data) => {
                        const node = {
                            data, selected: false, rowIndex: lib.nodes.length, isSelected: () => node.selected,
                            setData: (newData) => { node.data = newData; },
                        };
                        node.setSelected = (flag) => {
                            node.selected = flag;
                            lib.pending.push(() => options.onRowSelected({ node, data }));
                        };
                        return node;
                    });
                }
            },
            forEachNode: (fn) => lib.nodes.forEach(fn),
            getAllDisplayedColumns: () => (options.columnDefs || []).map((def) => ({ getColId: () => def.field })),
            getDisplayedRowCount: () => lib.nodes.length,
            refreshCells: jest_fn(),
            ensureNodeVisible: jest_fn(),
            getDisplayedRowAtIndex: (i) => lib.nodes[i],
            destroy: jest_fn(),
        };
        lib.api = api;
        return api;
    };
    lib.themeQuartz = { withParams: (params) => ({ params }) };
    return lib;
}
function jest_fn() {
    const fn = (...args) => { fn.calls.push(args); };
    fn.calls = [];
    return fn;
}

describe('createTableView', () => {
    const data = { columns: ['title'], rows: [{ index: 0, name: 'a', title: 'A' }, { index: 1, name: 'b', title: 'B' }] };

    it('selects the current page without telling the page to load it', () => {
        const lib = fakeGridLib();
        const onSelect = jest_fn();
        const view = createTableView({ gridEl: gridEl(), agGridLib: lib, onSelect });

        view.setData(data, 1);
        lib.flush();

        expect(lib.nodes.map((n) => n.selected)).toEqual([false, true]);
        expect(onSelect.calls).toEqual([]);
        expect(lib.api.ensureNodeVisible.calls.length).toBe(1);
    });

    it('moves the selection with the page and reports a row the user picks', () => {
        const lib = fakeGridLib();
        const onSelect = jest_fn();
        const view = createTableView({ gridEl: gridEl(), agGridLib: lib, onSelect });
        view.setData(data, 0);

        lib.flush();
        view.setCurrent(1);
        lib.flush();
        expect(lib.nodes.map((n) => n.selected)).toEqual([false, true]);
        expect(onSelect.calls).toEqual([]);

        lib.nodes[0].setSelected(true); // the user clicks the first row
        lib.flush();
        expect(onSelect.calls).toEqual([[0]]);
    });

    it('selects no row when the current page is not in the table', () => {
        const lib = fakeGridLib();
        const view = createTableView({ gridEl: gridEl(), agGridLib: lib, onSelect: jest_fn() });

        view.setData(data, -1);

        expect(lib.nodes.map((n) => n.selected)).toEqual([false, false]);
    });

    it('selects the row the focus moves to', () => {
        vi.useFakeTimers();
        const lib = fakeGridLib();
        const onSelect = jest_fn();
        const view = createTableView({ gridEl: gridEl(), agGridLib: lib, onSelect, focusDelayMs: 10 });
        view.setData(data, 0);

        lib.flush();
        lib.options.onCellFocused({ rowIndex: 1 });
        vi.advanceTimersByTime(20);
        lib.flush();

        expect(lib.nodes[1].selected).toBe(true);
        expect(onSelect.calls).toEqual([[1]]);
        vi.useRealTimers();
    });
});

describe('cellEditorFor', () => {
    it('edits one line of text in the cell', () => {
        expect(cellEditorFor({ value: 'one line' })).toMatchObject({ component: 'agTextCellEditor' });
        expect(cellEditorFor({ value: undefined })).toMatchObject({ component: 'agTextCellEditor' });
    });

    it('uses a larger box for text with line breaks, so they are not lost', () => {
        for (const value of ['a\nb', 'a\r\nb']) {
            expect(cellEditorFor({ value })).toMatchObject({ component: 'agLargeTextCellEditor', popup: true });
        }
    });
});

describe('parseClipboard and formatClipboard', () => {
    it('reads tab-separated cells and lines, ignoring one trailing line break', () => {
        expect(parseClipboard('a\tb\nc\td\n')).toEqual([['a', 'b'], ['c', 'd']]);
        expect(parseClipboard('one')).toEqual([['one']]);
        expect(parseClipboard('a\r\nb')).toEqual([['a'], ['b']]);
        expect(parseClipboard('')).toEqual([]);
    });

    it('reads a quoted cell that holds tabs, line breaks and quotes', () => {
        expect(parseClipboard('"line 1\nline 2"\t"say ""hi"""\tplain')).toEqual([['line 1\nline 2', 'say "hi"', 'plain']]);
    });

    it('keeps an empty cell between two tabs', () => {
        expect(parseClipboard('a\t\tc')).toEqual([['a', '', 'c']]);
    });

    it('formats what it reads', () => {
        const rows = [['plain', 'two\nlines'], ['say "hi"', '']];
        expect(parseClipboard(formatClipboard(rows))).toEqual(rows);
    });
});

describe('pasteTargets', () => {
    const bounds = rangeBounds({ anchor: { row: 2, col: 3 }, focus: { row: 1, col: 1 } });

    it('takes the corners of a range dragged in any direction', () => {
        expect(bounds).toEqual({ r1: 1, r2: 2, c1: 1, c2: 3 });
    });

    it('fills every cell of the range with a single value', () => {
        const targets = pasteTargets([['x']], bounds, 10, 10);
        expect(targets).toHaveLength(6);
        expect(new Set(targets.map((t) => t.text))).toEqual(new Set(['x']));
    });

    it('lays a block from the top-left of the range, cut off at the table edge', () => {
        const targets = pasteTargets([['a', 'b'], ['c', 'd']], bounds, 3, 2);
        expect(targets).toEqual([{ row: 1, col: 1, text: 'a' }, { row: 2, col: 1, text: 'c' }]);
    });
});

describe('pasting into the table', () => {
    const data = {
        columns: ['title', 'notes'],
        rows: [
            { index: 0, name: 'a', title: 'A', notes: 'n0', __editable: ['title', 'notes'] },
            { index: 1, name: 'b', title: 'B', notes: 'n1', __editable: ['title'] },
            { index: 2, name: 'c', title: 'C', __editable: ['title'] },
        ],
    };

    function setup(result) {
        const lib = fakeGridLib();
        const onEdit = vi.fn(async (edits) => result || { saved: edits.map(({ index, label }) => ({ index, label })), skipped: [] });
        const onStatus = jest_fn();
        const view = createTableView({ gridEl: gridEl(), agGridLib: lib, onSelect: jest_fn(), onEdit, onStatus });
        view.setData(data, 0);
        return { lib, view, onEdit, onStatus };
    }

    it('saves one pasted value to every editable cell in the range', async () => {
        const { lib, view, onEdit, onStatus } = setup();
        view.setRange({ anchor: { row: 0, col: 1 }, focus: { row: 2, col: 2 } }); // title + notes of all 3 pages

        view.pasteText('NEW');
        await new Promise((r) => setTimeout(r));

        const edits = onEdit.mock.calls[0][0];
        expect(edits.map((e) => [e.index, e.label])).toEqual([[0, 'title'], [0, 'notes'], [1, 'title'], [2, 'title']]);
        expect(new Set(edits.map((e) => e.text))).toEqual(new Set(['NEW']));
        expect(lib.nodes.map((n) => n.data.title)).toEqual(['NEW', 'NEW', 'NEW']);
        expect(onStatus.calls.at(-1)[0]).toBe('Saved 4 cells \u00b7 skipped 2 that cannot be edited'); // notes of pages 1 and 2
    });

    it('does not save a cell that already holds the text', async () => {
        const { view, onEdit } = setup();
        view.setRange({ anchor: { row: 0, col: 1 }, focus: { row: 0, col: 1 } });

        view.pasteText('A');
        await new Promise((r) => setTimeout(r));

        expect(onEdit).not.toHaveBeenCalled();
    });

    it('leaves a cell as it was when it could not be saved', async () => {
        const { lib, view } = setup({ saved: [], skipped: [{ index: 0, label: 'title', reason: 'no' }] });
        view.setRange({ anchor: { row: 0, col: 1 }, focus: { row: 0, col: 1 } });

        view.pasteText('NEW');
        await new Promise((r) => setTimeout(r));

        expect(lib.nodes[0].data.title).toBe('A');
    });

    it('never writes to the file name column', async () => {
        const { onEdit, view } = setup();
        view.setRange({ anchor: { row: 0, col: 0 }, focus: { row: 0, col: 0 } });

        view.pasteText('NEW');
        await new Promise((r) => setTimeout(r));

        expect(onEdit).not.toHaveBeenCalled();
    });
});

describe('the table tab in a saved arrangement', () => {
    it('is saved and restored like a DAG tab, once per panel', () => {
        const panel = new Panel();
        panel.addTab(new Tab({ id: 't', nodeId: null, label: 'Table', kind: 'table' }));

        const saved = serializeArrangement({ bottom: panel }, 'bottom', (id) => id);
        expect(saved.panels.bottom.tabs).toEqual([{ kind: 'table' }]);

        const restored = new Panel();
        let n = 0;
        restorePanelTabs(restored, { active: 0, tabs: [{ kind: 'table' }, { kind: 'table' }] }, {
            makeId: () => 'id' + n++, instancesFor: () => [], labelFor: (g) => g,
        });
        expect(restored.tabs.map((t) => [t.kind, t.label])).toEqual([['table', 'Table']]);
    });
});
