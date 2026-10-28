// The table view: one row per loaded page, one column per text artefact.
// Cells can be selected as a range, copied, and pasted into; a pasted value
// is saved to every cell it lands on. The helpers before createTableView are
// pure and unit tested in tests-js/; createTableView wraps AG Grid
// (vendor/ag-grid-community). The range, copy and paste are done here, not by
// AG Grid, because they are Enterprise features of it.

// Column definitions: the page's name, pinned on the left, then one column
// per text artefact label. humanize(label) is the header text. ctx (from the
// view) says which cells are in the selected range and which can be edited.
function tableColumnDefs(columns, humanize, ctx = {}) {
    return [
        { field: 'name', headerName: 'File', pinned: 'left', width: 200, minWidth: 120, cellClass: 'table-file-cell', cellClassRules: rangeClass(ctx) },
        ...columns.map((label) => ({
            field: label,
            headerName: humanize(label),
            tooltipField: label,
            flex: 1,
            minWidth: 180,
            cellClass: 'table-text-cell',
            cellClassRules: rangeClass(ctx),
            editable: ctx.isEditable ? (params) => ctx.isEditable(params.data, label) : false,
            cellEditorSelector: cellEditorFor,
        })),
    ];
}

// Text is edited in the cell itself. A single-line input would drop the line
// breaks of a longer text on saving, so text with any gets a larger box
// instead.
function cellEditorFor(params) {
    if (/[\r\n]/.test(params.value == null ? '' : String(params.value))) {
        return {
            component: 'agLargeTextCellEditor',
            popup: true,
            params: { maxLength: 1000000, rows: 12, cols: 60 },
        };
    }
    return { component: 'agTextCellEditor', params: { maxLength: 1000000 } };
}

function rangeClass(ctx) {
    return ctx.inRange ? { 'table-cell-in-range': (params) => ctx.inRange(params) } : {};
}

// The row for a page index, or null when there is none to select.
function tableRowIndex(rows, index) {
    return rows.some((row) => row.index === index) ? index : null;
}

// Splits clipboard text into rows of cells: tab-separated cells, one row per
// line, with "quoted" cells (as spreadsheets copy them) holding tabs, line
// breaks and doubled quotes. One trailing line break is ignored.
function parseClipboard(text) {
    const rows = [[]];
    let cell = '';
    let quoted = false;
    let started = false; // the current cell has had any character
    const endCell = () => { rows[rows.length - 1].push(cell); cell = ''; started = false; };
    for (let i = 0; i < text.length; i++) {
        const ch = text[i];
        if (quoted) {
            if (ch === '"' && text[i + 1] === '"') { cell += '"'; i++; }
            else if (ch === '"') quoted = false;
            else cell += ch;
        } else if (ch === '"' && !started) {
            quoted = true;
            started = true;
        } else if (ch === '\t') {
            endCell();
        } else if (ch === '\n' || ch === '\r') {
            if (ch === '\r' && text[i + 1] === '\n') i++;
            endCell();
            rows.push([]);
        } else {
            cell += ch;
            started = true;
        }
    }
    if (rows[rows.length - 1].length || cell || started) endCell();
    else rows.pop();
    return rows;
}

// The opposite: cells to clipboard text, quoting what needs it.
function formatClipboard(rows) {
    const quote = (value) => (/[\t\n\r"]/.test(value) ? `"${value.replace(/"/g, '""')}"` : value);
    return rows.map((row) => row.map(quote).join('\t')).join('\n');
}

// The corners of a range, whichever way it was dragged.
function rangeBounds(range) {
    if (!range) return null;
    return {
        r1: Math.min(range.anchor.row, range.focus.row),
        r2: Math.max(range.anchor.row, range.focus.row),
        c1: Math.min(range.anchor.col, range.focus.col),
        c2: Math.max(range.anchor.col, range.focus.col),
    };
}

// Which cells a paste lands on. One value fills every cell of the range; a
// block of values lands with its top-left cell on the range's top-left cell,
// cut off at the edge of the table.
function pasteTargets(matrix, bounds, rowCount, colCount) {
    const targets = [];
    if (!matrix.length || !bounds) return targets;
    if (matrix.length === 1 && matrix[0].length === 1) {
        for (let row = bounds.r1; row <= bounds.r2; row++) {
            for (let col = bounds.c1; col <= bounds.c2; col++) targets.push({ row, col, text: matrix[0][0] });
        }
        return targets;
    }
    matrix.forEach((cells, dr) => cells.forEach((text, dc) => {
        const row = bounds.r1 + dr;
        const col = bounds.c1 + dc;
        if (row < rowCount && col < colCount) targets.push({ row, col, text });
    }));
    return targets;
}

// Colours come from the app's own theme variables, so the grid follows
// whichever theme.css the project uses.
function tableTheme(agGridLib) {
    return agGridLib.themeQuartz.withParams({
        backgroundColor: 'var(--app-bg, #002b36)',
        foregroundColor: 'var(--app-fg, #93a1a1)',
        headerBackgroundColor: 'var(--app-surface, #073642)',
        headerTextColor: 'var(--app-fg-bright, #fdf6e3)',
        borderColor: 'var(--app-border, #586e75)',
        accentColor: 'var(--app-primary, #268bd2)',
        selectedRowBackgroundColor: 'rgba(var(--app-primary-rgb, 38, 139, 210), 0.35)',
        rowHoverColor: 'rgba(var(--app-primary-rgb, 38, 139, 210), 0.15)',
        fontFamily: 'var(--app-font-family, inherit)',
        fontSize: 13,
    });
}

function isEditableCell(data, label) {
    return !!data && Array.isArray(data.__editable) && data.__editable.includes(label);
}

// Builds the grid in gridEl.
//  onSelect(index)  the user moved to another row (a click, or the arrow keys
//                   once the focus is in the grid); selecting a row from code
//                   (setCurrent) does not call it.
//  onEdit(edits)    saves [{index, label, text}]; resolves to
//                   {saved: [{index, label}], skipped: [{index, label, reason}]}.
//  onStatus(text)   a line to tell the user what a paste did.
function createTableView({
    gridEl, agGridLib = window.agGrid, humanize = (label) => label,
    onSelect, onEdit, onStatus, focusDelayMs = 150,
}) {
    let focusTimer = null;
    let rows = [];
    let range = null; // { anchor: {row, col}, focus: {row, col} }, in displayed rows and columns
    let dragging = false;
    // The page whose row is selected. AG Grid reports a selection some time
    // after it is made, so a selection made from code is told apart by being
    // the current page, not by a flag around the call.
    let current = null;

    const ctx = {
        isEditable: isEditableCell,
        inRange: (params) => {
            const bounds = rangeBounds(range);
            if (!bounds || params.node.rowIndex == null) return false;
            const col = columns().indexOf(params.column);
            return params.node.rowIndex >= bounds.r1 && params.node.rowIndex <= bounds.r2
                && col >= bounds.c1 && col <= bounds.c2;
        },
    };
    const columns = () => api.getAllDisplayedColumns();
    const rowCount = () => api.getDisplayedRowCount();

    const select = (index) => {
        current = tableRowIndex(rows, index);
        api.forEachNode((node) => {
            const selected = node.data.index === current;
            if (node.isSelected() !== selected) node.setSelected(selected);
            if (selected) api.ensureNodeVisible(node, 'middle');
        });
    };

    const setRange = (next) => {
        range = next;
        api.refreshCells({ force: true });
    };
    const cellAt = (event) => ({ row: event.node.rowIndex, col: columns().indexOf(event.column) });
    const isRangeSingle = () => {
        const b = rangeBounds(range);
        return !b || (b.r1 === b.r2 && b.c1 === b.c2);
    };

    const valueAt = (row, col) => {
        const node = api.getDisplayedRowAtIndex(row);
        const column = columns()[col];
        if (!node || !column) return '';
        const value = node.data[column.getColId()];
        return value == null ? '' : String(value);
    };

    // Writes text to the cells, and lets the user know what was saved.
    const apply = async (targets) => {
        const edits = [];
        let readOnly = 0;
        for (const { row, col, text } of targets) {
            const node = api.getDisplayedRowAtIndex(row);
            const column = columns()[col];
            const label = column && column.getColId();
            if (!node || label === 'name' || !isEditableCell(node.data, label)) { readOnly++; continue; }
            if (valueAt(row, col) !== text) edits.push({ index: node.data.index, label, text });
        }
        let result = { saved: [], skipped: [] };
        if (edits.length && onEdit) result = await onEdit(edits);
        const byKey = new Map(edits.map((edit) => [`${edit.index}\u0000${edit.label}`, edit.text]));
        for (const { index, label } of result.saved) {
            api.forEachNode((node) => {
                if (node.data.index !== index) return;
                node.setData({ ...node.data, [label]: byKey.get(`${index}\u0000${label}`) });
            });
        }
        if (onStatus) {
            const skipped = readOnly + result.skipped.length;
            const parts = [`Saved ${result.saved.length} cell${result.saved.length === 1 ? '' : 's'}`];
            if (skipped) parts.push(`skipped ${skipped} that cannot be edited`);
            onStatus(parts.join(' · '));
        }
    };

    const rangeText = () => {
        const b = rangeBounds(range);
        if (!b) return '';
        const lines = [];
        for (let row = b.r1; row <= b.r2; row++) {
            const cells = [];
            for (let col = b.c1; col <= b.c2; col++) cells.push(valueAt(row, col));
            lines.push(cells);
        }
        return formatClipboard(lines);
    };

    const inEditor = (event) => /^(INPUT|TEXTAREA)$/.test(event.target.tagName) || !!event.target.closest('.ag-popup');
    const onCopy = (event) => {
        if (inEditor(event) || !range) return;
        event.preventDefault();
        event.clipboardData.setData('text/plain', rangeText());
    };
    const onPaste = (event) => {
        if (inEditor(event) || !range) return;
        const text = event.clipboardData ? event.clipboardData.getData('text/plain') : '';
        event.preventDefault();
        const targets = pasteTargets(parseClipboard(text), rangeBounds(range), rowCount(), columns().length);
        if (targets.length) apply(targets);
    };
    const onMouseUp = () => { dragging = false; };
    gridEl.addEventListener('copy', onCopy);
    gridEl.addEventListener('paste', onPaste);
    window.addEventListener('mouseup', onMouseUp);

    const api = agGridLib.createGrid(gridEl, {
        theme: tableTheme(agGridLib),
        columnDefs: tableColumnDefs([], humanize, ctx),
        rowData: [],
        getRowId: (params) => String(params.data.index),
        suppressFieldDotNotation: true, // labels may contain dots
        rowSelection: { mode: 'singleRow', checkboxes: false, enableClickSelection: false },
        tooltipShowDelay: 400,
        onRowSelected: (event) => {
            if (!event.node.isSelected() || event.data.index === current) return;
            current = event.data.index;
            if (onSelect) onSelect(current);
        },
        // Pressing on a cell starts a range; Shift extends it from where it began.
        onCellMouseDown: (event) => {
            if (event.event.button !== 0) return;
            const cell = cellAt(event);
            if (event.event.shiftKey && range) setRange({ anchor: range.anchor, focus: cell });
            else setRange({ anchor: cell, focus: cell });
            dragging = true;
        },
        onCellMouseOver: (event) => {
            if (dragging && range) setRange({ anchor: range.anchor, focus: cellAt(event) });
        },
        // A plain click (not a range) opens that page.
        onCellClicked: (event) => {
            if (event.event.shiftKey || !isRangeSingle()) return;
            if (!event.node.isSelected()) event.node.setSelected(true);
        },
        // Moving the focus with the arrow keys moves the selection with it.
        onCellFocused: (event) => {
            if (event.rowIndex == null || (event.sourceEvent && event.sourceEvent.type !== 'keydown')) return;
            setRange({ anchor: { row: event.rowIndex, col: columns().indexOf(event.column) },
                       focus: { row: event.rowIndex, col: columns().indexOf(event.column) } });
            clearTimeout(focusTimer);
            focusTimer = setTimeout(() => {
                const node = api.getDisplayedRowAtIndex(event.rowIndex);
                if (node && !node.isSelected()) node.setSelected(true);
            }, focusDelayMs);
        },
        // Typing into one cell (double-click) saves like a paste does.
        onCellValueChanged: async (event) => {
            if (event.newValue === event.oldValue || !onEdit) return;
            const label = event.colDef.field;
            const result = await onEdit([{ index: event.data.index, label, text: event.newValue == null ? '' : String(event.newValue) }]);
            if (!result.saved.length) event.node.setData({ ...event.data, [label]: event.oldValue }); // not saved: put it back
            if (onStatus) onStatus(result.saved.length ? 'Saved 1 cell' : 'Could not save that cell');
        },
    });

    return {
        api,
        // Replaces the data and highlights the current page's row.
        setData({ columns: labels, rows: newRows }, currentIndex) {
            rows = newRows;
            range = null;
            api.setGridOption('columnDefs', tableColumnDefs(labels, humanize, ctx));
            api.setGridOption('rowData', rows);
            select(currentIndex);
        },
        setCurrent(index) {
            select(index);
        },
        // For tests and callers that move the range from code.
        setRange,
        pasteText(text) {
            if (range) apply(pasteTargets(parseClipboard(text), rangeBounds(range), rowCount(), columns().length));
        },
        destroy() {
            clearTimeout(focusTimer);
            gridEl.removeEventListener('copy', onCopy);
            gridEl.removeEventListener('paste', onPaste);
            window.removeEventListener('mouseup', onMouseUp);
            api.destroy();
        },
    };
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        tableColumnDefs, tableRowIndex, tableTheme, createTableView, cellEditorFor,
        parseClipboard, formatClipboard, rangeBounds, pasteTargets, isEditableCell,
    };
}
