const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync(require.resolve('../collectra/gui/index.html'), 'utf8');
const runDialog = require('../collectra/gui/runDialog.js');

const tagAt = (id) => {
    const at = html.indexOf(`id="${id}"`);
    return {start: html.lastIndexOf('<', at), at};
};
const at = html.indexOf('id="runToolbarBtn"');
const button = html.slice(html.lastIndexOf('<button', at), html.indexOf('</button>', at));
const openTag = button.slice(0, button.indexOf('>') + 1);
const modalStart = html.indexOf('<div id="runModal"');
const modal = html.slice(modalStart, html.indexOf('<!--', modalStart));

describe('the Run toolbar button', () => {
    it('sits right after the search button and right before the DAG button', () => {
        const search = tagAt('artefactSearchToggle');
        const searchEnd = html.indexOf('</button>', search.at) + '</button>'.length;
        const run = tagAt('runToolbarBtn');
        expect(html.slice(searchEnd, run.start).trim()).toBe('');
        const runEnd = html.indexOf('</button>', run.at) + '</button>'.length;
        expect(html.slice(runEnd, tagAt('dagToolbarBtn').start).trim()).toBe('');
    });

    it('is a normal toolbar button with its labels, disabled until a page is loaded', () => {
        expect(openTag).toContain('class="panel-toolbar-btn"');
        expect(openTag).toContain('onclick="showRunDialog()"');
        expect(openTag).toContain('title="Run"');
        expect(openTag).toContain('aria-label="Run pipeline on this page"');
        expect(openTag).toMatch(/\sdisabled>$/);
    });

    it('shows a play triangle the same size and colour as the other icons', () => {
        expect(button).toContain('data-icon="play"');
        expect(button).toContain('width="16" height="16" viewBox="0 0 16 16"');
        expect(button).toContain('fill="none" stroke="currentColor"');
        expect((button.match(/<path /g) || []).length).toBe(1);
    });

    it('is enabled at the same point as the Tasks button', () => {
        expect(html).toContain(
            "document.getElementById('tasksToolbarBtn').disabled = false;\n"
            + "                document.getElementById('runToolbarBtn').disabled = false;");
    });

    it('loads the panel content script next to the delete dialog script', () => {
        expect(html).toContain('<script src="deleteDialog.js"></script>\n    <script src="runDialog.js"></script>');
    });
});

describe('the Run confirmation panel markup', () => {
    it('reuses the delete dialog markup and classes', () => {
        expect(modal).toContain('class="modal-overlay" role="dialog" aria-modal="true" aria-labelledby="runModalTitle"');
        expect(modal).toContain('<div class="modal-popup">');
        expect(modal).toContain('<h3 id="runModalTitle"></h3>');
        expect(modal).toContain('<ul id="runModalTasks" class="delete-children" hidden></ul>');
        expect(modal).toContain('<p id="runModalSummary" class="delete-summary" hidden></p>');
    });

    it('has a single-line Note box of at most 200 characters', () => {
        const note = modal.slice(modal.lastIndexOf('<input', modal.indexOf('id="runNoteInput"')));
        const tag = note.slice(0, note.indexOf('>') + 1);
        expect(tag).toContain('type="text"');
        expect(tag).toContain('maxlength="200"');
        expect(modal).toContain('<label class="modal-field-label" for="runNoteInput">Note</label>');
        expect(modal).not.toContain('<textarea');
    });

    it('has Cancel and Run buttons', () => {
        expect(modal).toContain('<button class="btn-cancel" id="runModalCancel" onclick="closeRunDialog()">Cancel</button>');
        expect(modal).toContain('<button class="btn-confirm" id="runModalConfirm" onclick="startRun()">Run</button>');
    });

    it('has a progress popover with a Stop button, styled like the toolbar popovers', () => {
        const start = html.indexOf('<div class="lineage-popover" id="runProgressPopover"');
        expect(start).toBeGreaterThan(-1);
        const popover = html.slice(start, html.indexOf('<div class="panels-row">', start));
        expect(popover).toContain('<div class="lineage-popover-header">Run</div>');
        expect(popover).toContain('id="runProgressText"');
        expect(popover).toContain('<button class="btn-cancel" id="runStopBtn" onclick="stopRun()">Stop</button>');
        expect(popover).toContain('Editing this page is paused until the run finishes.');
    });
});

// ── Behaviour: the page's own run functions against stand-in elements ──

const source = html.slice(html.indexOf('// ── RUN BUTTON'), html.indexOf('// ── END RUN BUTTON'));

function element(tag = 'div') {
    const classes = new Set();
    const attributes = {};
    return {
        tagName: tag.toUpperCase(), value: '', textContent: '', hidden: false, disabled: false,
        focused: false, style: {}, children: [], className: '', title: '',
        classList: {
            add: (c) => classes.add(c), remove: (c) => classes.delete(c),
            toggle: (c, on) => { (on === undefined ? !classes.has(c) : on) ? classes.add(c) : classes.delete(c); },
            contains: (c) => classes.has(c),
        },
        setAttribute(name, value) { attributes[name] = String(value); },
        getAttribute(name) { return attributes[name]; },
        replaceChildren(...nodes) { this.children = nodes; },
        append(...nodes) { this.children.push(...nodes); },
        focus() { this.focused = true; },
    };
}

function page({preview, start = {success: true}, statuses = [], openYaml = '/data/p1/results.yaml', folderIndex = 0} = {}) {
    const ids = ['runToolbarBtn', 'runModal', 'runModalTitle', 'runModalSummary', 'runModalTasks', 'runNoteInput',
        'runModalError', 'runModalCancel', 'runModalConfirm', 'runProgressPopover', 'runProgressText',
        'runProgressPaused', 'runStopBtn', 'runCloseBtn'];
    const els = Object.fromEntries(ids.map((id) => [id, element()]));
    els.runCloseBtn.hidden = true;
    const queue = [...statuses];
    const api = {
        get_run_preview: vi.fn().mockResolvedValue(preview || {success: true, can_run: true, message: '',
            tasks: [{name: 'ocr', action: 'run', reason: 'no output yet'},
                {name: 'detect', action: 'skip', reason: 'already has output'}]}),
        start_run: vi.fn().mockResolvedValue(start),
        get_run_status: vi.fn(async () => (queue.length > 1 ? queue.shift() : queue[0])),
        stop_run: vi.fn().mockResolvedValue({success: true}),
    };
    const context = {
        ...runDialog,
        document: {getElementById: (id) => els[id], createElement: (tag) => element(tag), addEventListener() {}},
        window: {pywebview: {api}},
        selectedYamlPath: openYaml, currentFolderIndex: folderIndex, folderLoading: false,
        loadCollectraFolderInner: vi.fn().mockResolvedValue(undefined),
        loadYaml: vi.fn().mockResolvedValue(undefined),
        refreshTables: vi.fn(),
        panelControllers: {bottom: {showActiveTabContent: vi.fn()}},
        setTimeout, clearTimeout, console,
    };
    vm.createContext(context);
    vm.runInContext(source, context);
    return {els, api, context};
}

const running = (done, extra = {}) => ({running: true, finished: false, folder: '/data/p1', current: null,
    done, total: 2, summary: null, error: null, ...extra});
const finished = (extra = {}) => ({running: false, finished: true, folder: '/data/p1', current: null,
    done: 1, total: 1, summary: '1 ran, 1 skipped, 0 failed', error: null, ...extra});

beforeEach(() => vi.useFakeTimers());
afterEach(() => vi.useRealTimers());

it('opening the panel shows the preview, the note box empty and the Run label', async () => {
    const p = page();
    p.els.runNoteInput.value = 'left over';
    await p.context.showRunDialog();
    expect(p.els.runModal.classList.contains('visible')).toBe(true);
    expect(p.els.runModalTitle.textContent).toBe('Run the pipeline on this page?');
    expect(p.els.runModalTasks.children.map((li) => li.children[0].textContent))
        .toEqual(['ocr — will run (no output yet)', 'detect — skipped (already has output)']);
    expect(p.els.runModalTasks.hidden).toBe(false);
    expect(p.els.runModalSummary.textContent).toBe('Your edits are kept.');
    expect(p.els.runModalConfirm.textContent).toBe('Run 1 task');
    expect(p.els.runModalConfirm.disabled).toBe(false);
    expect(p.els.runNoteInput.value).toBe('');
});

it('shows the backend message and cannot run when the backend says no', async () => {
    const p = page({preview: {success: true, can_run: false, tasks: [], message: 'A run is already in progress'}});
    await p.context.showRunDialog();
    expect(p.els.runModalSummary.textContent).toBe('A run is already in progress');
    expect(p.els.runModalSummary.hidden).toBe(false);
    expect(p.els.runModalTasks.hidden).toBe(true);
    expect(p.els.runModalConfirm.disabled).toBe(true);
});

it('shows the warning when a task re-runs', async () => {
    const p = page({preview: {success: true, can_run: true, message: '', tasks: [
        {name: 'ocr', action: 'run', reason: 'no output yet'},
        {name: 'classify', action: 'rerun', reason: 'upstream ocr runs'}]}});
    await p.context.showRunDialog();
    expect(p.els.runModalSummary.textContent).toBe('Re-run tasks replace their existing output, including edits.');
});

it('Cancel closes the panel without running', async () => {
    const p = page();
    await p.context.showRunDialog();
    p.context.closeRunDialog();
    expect(p.els.runModal.classList.contains('visible')).toBe(false);
    expect(p.api.start_run).not.toHaveBeenCalled();
});

it('Run starts the run with the note, then polls and shows progress and a Stop button', async () => {
    const p = page({statuses: [running(0), running(1), finished()]});
    await p.context.showRunDialog();
    p.els.runNoteInput.value = '  retry after fix  ';
    await p.context.startRun();
    expect(p.api.start_run).toHaveBeenCalledWith('retry after fix');
    expect(p.els.runModal.classList.contains('visible')).toBe(false);
    expect(p.els.runProgressPopover.style.display).toBe('block');
    expect(p.els.runStopBtn.hidden).toBe(false);
    expect(p.els.runProgressPaused.hidden).toBe(false);
    expect(p.els.runToolbarBtn.classList.contains('active')).toBe(true);
    expect(p.els.runProgressText.textContent).toBe('Running… (0 of 2 done)');
    await vi.advanceTimersByTimeAsync(1000);
    expect(p.els.runProgressText.textContent).toBe('Running… (1 of 2 done)');
    await vi.advanceTimersByTimeAsync(1000);
    expect(p.els.runProgressText.textContent).toBe('Run finished: 1 ran, 1 skipped, 0 failed. See run-log.txt.');
    expect(p.els.runStopBtn.hidden).toBe(true);
    expect(p.els.runCloseBtn.hidden).toBe(false);
    expect(p.els.runProgressPaused.hidden).toBe(true);
    expect(p.els.runToolbarBtn.classList.contains('active')).toBe(false);
    const calls = p.api.get_run_status.mock.calls.length;
    await vi.advanceTimersByTimeAsync(5000);
    expect(p.api.get_run_status.mock.calls.length).toBe(calls); // polling stopped
});

it('Stop asks the backend to stop and the progress says it is stopping', async () => {
    const p = page({statuses: [running(0), running(1)]});
    await p.context.showRunDialog();
    await p.context.startRun();
    await p.context.stopRun();
    expect(p.api.stop_run).toHaveBeenCalled();
    expect(p.els.runStopBtn.disabled).toBe(true);
    await vi.advanceTimersByTimeAsync(1000);
    expect(p.els.runProgressText.textContent).toBe('Stopping after the current task… (1 of 2 done)');
});

it('a refused start keeps the panel open and shows the backend error', async () => {
    const p = page({start: {success: false, error: 'A Git operation is running; try again when it finishes'}});
    await p.context.showRunDialog();
    await p.context.startRun();
    expect(p.els.runModal.classList.contains('visible')).toBe(true);
    expect(p.els.runModalError.hidden).toBe(false);
    expect(p.els.runModalError.textContent).toBe('A Git operation is running; try again when it finishes');
    expect(p.els.runModalConfirm.disabled).toBe(false);
    expect(p.api.get_run_status).not.toHaveBeenCalled();
});

it('a start that throws shows its message', async () => {
    const p = page();
    p.api.start_run.mockRejectedValue(new Error('bridge gone'));
    await p.context.showRunDialog();
    await p.context.startRun();
    expect(p.els.runModalError.textContent).toBe('bridge gone');
});

it('a preview that throws shows its message and cannot run', async () => {
    const p = page();
    p.api.get_run_preview.mockRejectedValue(new Error('bridge gone'));
    await p.context.showRunDialog();
    expect(p.els.runModalSummary.textContent).toBe('bridge gone');
    expect(p.els.runModalConfirm.disabled).toBe(true);
});

it('completion reloads the open page when it is the run folder', async () => {
    const p = page({statuses: [finished()]});
    await p.context.showRunDialog();
    await p.context.startRun();
    expect(p.context.loadCollectraFolderInner).toHaveBeenCalledWith(0);
    expect(p.context.refreshTables).toHaveBeenCalled();
    expect(p.context.folderLoading).toBe(false);
});

it('completion does not reload when another page is now open', async () => {
    const p = page({statuses: [running(0), finished()]});
    await p.context.showRunDialog();
    await p.context.startRun();
    p.context.selectedYamlPath = '/data/p2/results.yaml';
    await vi.advanceTimersByTimeAsync(1000);
    expect(p.els.runProgressText.textContent).toBe('Run finished: 1 ran, 1 skipped, 0 failed. See run-log.txt.');
    expect(p.context.loadCollectraFolderInner).not.toHaveBeenCalled();
    expect(p.context.loadYaml).not.toHaveBeenCalled();
});

it('completion reloads a page opened on its own (no folder list) through its results file', async () => {
    const p = page({statuses: [finished()], folderIndex: -1});
    await p.context.showRunDialog();
    await p.context.startRun();
    expect(p.context.loadYaml).toHaveBeenCalled();
    expect(p.context.panelControllers.bottom.showActiveTabContent).toHaveBeenCalled();
    expect(p.context.loadCollectraFolderInner).not.toHaveBeenCalled();
});

it('shows a reload or log error from the backend after the summary', async () => {
    const p = page({statuses: [finished({error: 'The page could not be reloaded after the run: bad'})]});
    await p.context.showRunDialog();
    await p.context.startRun();
    expect(p.els.runProgressText.textContent)
        .toBe('Run finished: 1 ran, 1 skipped, 0 failed. See run-log.txt. The page could not be reloaded after the run: bad');
});

it('keeps polling when one status call fails, and says so', async () => {
    const p = page({statuses: [running(0), finished()]});
    p.api.get_run_status.mockRejectedValueOnce(new Error('busy'));
    await p.context.showRunDialog();
    await p.context.startRun();
    expect(p.els.runProgressText.textContent).toBe("Could not read the run's progress: busy");
    await vi.advanceTimersByTimeAsync(1000);
    expect(p.els.runProgressText.textContent).toBe('Running… (0 of 2 done)');
});

it('the toolbar button shows the progress instead of a new panel while a run is going', async () => {
    const p = page({statuses: [running(0)]});
    await p.context.showRunDialog();
    await p.context.startRun();
    p.context.closeRunProgress();
    expect(p.els.runProgressPopover.style.display).toBe('none');
    p.api.get_run_preview.mockClear();
    await p.context.showRunDialog();
    expect(p.api.get_run_preview).not.toHaveBeenCalled();
    expect(p.els.runProgressPopover.style.display).toBe('block');
    expect(p.els.runModal.classList.contains('visible')).toBe(false);
});
