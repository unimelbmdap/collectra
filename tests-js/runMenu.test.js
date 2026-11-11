// The menu under the Run button, the "Overwrite existing results" tick and the
// panel for a run of the whole collection.
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync(require.resolve('../collectra/gui/index.html'), 'utf8');
const runDialog = require('../collectra/gui/runDialog.js');

const openTagOf = (id) => {
    const at = html.indexOf(`id="${id}"`);
    return html.slice(html.lastIndexOf('<', at), html.indexOf('>', at) + 1);
};

describe('the markup', () => {
    it('opens the menu from the Run button', () => {
        expect(openTagOf('runToolbarBtn')).toContain('onclick="showRunMenu(event)"');
    });

    it('has the tick in the panel, marked up like the "Override shell" tick', () => {
        const modalStart = html.indexOf('<div id="runModal"');
        const modal = html.slice(modalStart, html.indexOf('<!--', modalStart));
        expect(modal).toMatch(/<label title="[^"]+"><input type="checkbox" id="runOverwriteInput" onchange="toggleRunOverwrite\(this\.checked\)"> Overwrite existing results<\/label>/);
    });
});

const source = html.slice(html.indexOf('// ── RUN BUTTON'), html.indexOf('// ── END RUN BUTTON'));

function element(tag = 'div') {
    const classes = new Set();
    return {
        tagName: tag.toUpperCase(), value: '', textContent: '', hidden: false, disabled: false, checked: false,
        focused: false, style: {}, children: [], className: '', title: '', dataset: {},
        classList: {
            add: (c) => classes.add(c), remove: (c) => classes.delete(c),
            toggle: (c, on) => { (on === undefined ? !classes.has(c) : on) ? classes.add(c) : classes.delete(c); },
            contains: (c) => classes.has(c),
        },
        replaceChildren(...nodes) { this.children = nodes; },
        append(...nodes) { this.children.push(...nodes); },
        appendChild(node) { this.children.push(node); },
        focus() { this.focused = true; },
    };
}

const collectionPreview = (extra = {}) => ({success: true, can_run: true, pages: 3, to_replace: 1, first_time: 1,
    tasks: [{name: 'ocr', pages: 2, replace: 0}], unplannable: [{name: 'p9', reason: 'no image'}], message: '', ...extra});

function page({statuses = [], openYaml = '/data/p1/results.yaml'} = {}) {
    const ids = ['runToolbarBtn', 'runModal', 'runModalTitle', 'runModalSummary', 'runModalTasks', 'runNoteInput',
        'runModalError', 'runModalCancel', 'runModalConfirm', 'runProgressPopover', 'runProgressText',
        'runProgressPaused', 'runStopBtn', 'runCloseBtn', 'runPausedNotice', 'runOverwriteInput'];
    const els = Object.fromEntries(ids.map((id) => [id, element()]));
    const queue = [...statuses];
    const api = {
        get_run_preview: vi.fn(async (overwrite) => ({success: true, can_run: true, message: '',
            tasks: [{name: 'ocr', action: overwrite ? 'rerun' : 'skip', reason: overwrite ? 'overwrite' : 'already has output'},
                {name: 'detect', action: 'run', reason: 'no output yet'}]})),
        get_collection_preview: vi.fn().mockResolvedValue(collectionPreview()),
        start_run: vi.fn().mockResolvedValue({success: true}),
        start_collection_run: vi.fn().mockResolvedValue({success: true}),
        get_run_status: vi.fn(async () => (queue.length > 1 ? queue.shift() : queue[0])),
        stop_run: vi.fn().mockResolvedValue({success: true}),
    };
    const context = {
        ...runDialog,
        document: {getElementById: (id) => els[id], createElement: (tag) => element(tag)},
        window: {pywebview: {api}},
        buildArrangePopover: vi.fn(() => { els.nodeContextMenu = element(); }),
        closeNodeContextMenu: vi.fn(() => { delete els.nodeContextMenu; }),
        selectedYamlPath: openYaml, currentFolderIndex: 0, folderLoading: false,
        loadCollectraFolderInner: vi.fn().mockResolvedValue(undefined),
        refreshTables: vi.fn(),
        panelControllers: {bottom: {setTextRunLocked: vi.fn(), setImageRunLocked: vi.fn()}},
        setTimeout, clearTimeout, console,
    };
    vm.createContext(context);
    vm.runInContext(source, context);
    const menu = () => {
        context.showRunMenu({currentTarget: els.runToolbarBtn});
        return els.nodeContextMenu;
    };
    const choose = (label) => menu().children.find((item) => item.textContent === label).onclick();
    const shown = () => els.runModalTasks.children.map((li) => li.children[0].textContent);
    return {els, api, context, menu, choose, shown};
}

beforeEach(() => vi.useFakeTimers());
afterEach(() => vi.useRealTimers());

it('the Run button opens a Split View style menu with the two entries', () => {
    const p = page();
    const items = p.menu().children;
    expect(p.context.buildArrangePopover).toHaveBeenCalledWith(p.els.runToolbarBtn, 'Run', []);
    expect(items.map((item) => item.textContent)).toEqual(['Run this image', 'Run whole collection']);
    expect(items.every((item) => item.className === 'layout-menu-item')).toBe(true);
});

it('"Run this image" closes the menu and opens the panel for the open page, unchanged when unticked', async () => {
    const p = page();
    await p.choose('Run this image');
    expect(p.context.closeNodeContextMenu).toHaveBeenCalled();
    expect(p.api.get_run_preview).toHaveBeenCalledWith(false);
    expect(p.els.runModal.classList.contains('visible')).toBe(true);
    expect(p.els.runModalTitle.textContent).toBe('Run the pipeline on this page?');
    expect(p.shown()).toEqual(['ocr — skipped (already has output)', 'detect — will run (no output yet)']);
    expect(p.els.runModalSummary.textContent).toBe('Your edits are kept.');
    expect(p.els.runModalConfirm.textContent).toBe('Run 1 task');
});

it('"Run whole collection" opens the panel with the collection preview', async () => {
    const p = page();
    await p.choose('Run whole collection');
    expect(p.api.get_collection_preview).toHaveBeenCalledWith(false);
    expect(p.api.get_run_preview).not.toHaveBeenCalled();
    expect(p.els.runModal.classList.contains('visible')).toBe(true);
    expect(p.els.runModalTitle.textContent).toBe('Run the whole collection?');
    expect(p.els.runModalSummary.textContent)
        .toBe('3 pages: 1 have results, 1 have none yet, 1 cannot run. Nothing existing is replaced.');
    expect(p.shown()).toEqual(['ocr — runs on 2 pages', 'p9 — cannot run (no image)']);
    expect(p.els.runModalConfirm.textContent).toBe('Run 2 pages');
    expect(p.els.runModalConfirm.disabled).toBe(false);
});

it('shows the backend message and cannot run when the collection cannot run', async () => {
    const p = page();
    p.api.get_collection_preview.mockResolvedValue(collectionPreview({can_run: false, message: 'No page can run'}));
    await p.choose('Run whole collection');
    expect(p.els.runModalSummary.textContent).toBe('No page can run');
    expect(p.els.runModalConfirm.disabled).toBe(true);
});

it('the tick is off whenever a panel opens, also after closing and reopening', async () => {
    const p = page({statuses: [{running: false, finished: true, folder: '', folders: [], unit: 'pages', current: null,
        done: 0, total: 0, summary: '0 pages: 0 ran', error: null}]});
    p.els.runOverwriteInput.checked = true;
    await p.choose('Run this image');
    expect(p.els.runOverwriteInput.checked).toBe(false);
    p.els.runOverwriteInput.checked = true;
    await p.context.toggleRunOverwrite(true);
    p.context.closeRunDialog();
    await p.choose('Run whole collection');
    expect(p.els.runOverwriteInput.checked).toBe(false);
    expect(p.api.get_collection_preview).toHaveBeenLastCalledWith(false);
    await p.context.startRun();
    expect(p.api.start_collection_run).toHaveBeenCalledWith('');
});

it('toggling the tick fetches the preview again and says what is replaced, for one image', async () => {
    const p = page();
    await p.choose('Run this image');
    p.els.runNoteInput.value = 'typed';
    await p.context.toggleRunOverwrite(true);
    expect(p.api.get_run_preview).toHaveBeenLastCalledWith(true);
    expect(p.shown()).toEqual(['ocr — will re-run (overwrite)', 'detect — will run (no output yet)']);
    expect(p.els.runModalSummary.textContent).toBe('All 2 tasks will run again on this image and replace their '
        + 'results, including edits and annotations you made.');
    expect(p.els.runModalConfirm.textContent).toBe('Run 2 tasks');
    expect(p.els.runNoteInput.value).toBe('typed');
    await p.context.toggleRunOverwrite(false);
    expect(p.api.get_run_preview).toHaveBeenLastCalledWith(false);
    expect(p.els.runModalSummary.textContent).toBe('Your edits are kept.');
});

it('toggling the tick fetches the collection preview again', async () => {
    const p = page();
    await p.choose('Run whole collection');
    p.api.get_collection_preview.mockResolvedValue(collectionPreview({tasks: [{name: 'ocr', pages: 2, replace: 1}]}));
    await p.context.toggleRunOverwrite(true);
    expect(p.api.get_collection_preview).toHaveBeenLastCalledWith(true);
    expect(p.shown()[0]).toBe('ocr — runs on 2 pages (replaces results on 1 page)');
    expect(p.els.runModalSummary.textContent).toBe('3 pages: 1 have results, 1 have none yet, 1 cannot run. '
        + 'All tasks will run again on 2 pages and replace their results, including edits and annotations you made.');
});

it('starting passes the note and the tick, for one image and for the collection', async () => {
    const p = page({statuses: [{running: false, finished: true, folder: '/data/p1', current: null, done: 1,
        total: 1, summary: '1 ran', error: null}]});
    await p.choose('Run this image');
    await p.context.toggleRunOverwrite(true);
    p.els.runNoteInput.value = ' again ';
    await p.context.startRun();
    expect(p.api.start_run).toHaveBeenCalledWith('again', true);
    await p.choose('Run whole collection');
    await p.context.toggleRunOverwrite(true);
    p.els.runNoteInput.value = 'all';
    await p.context.startRun();
    expect(p.api.start_collection_run).toHaveBeenCalledWith('all', true);
});

it('inserts backend text as text only', async () => {
    const p = page();
    p.api.get_collection_preview.mockResolvedValue(collectionPreview({tasks: [{name: '<b>x</b>', pages: 1, replace: 0}],
        unplannable: [{name: '<i>p</i>', reason: '<img src=x>'}]}));
    await p.choose('Run whole collection');
    expect(p.shown()).toEqual(['<b>x</b> — runs on 1 page', '<i>p</i> — cannot run (<img src=x>)']);
});

describe('a run of the collection', () => {
    const status = (extra = {}) => ({running: true, finished: false, folder: '', folders: ['/data/p1', '/data/p2'],
        unit: 'pages', current: null, done: 1, total: 2, summary: null, error: null, ...extra});
    const done = status({running: false, finished: true, done: 2, summary: '2 pages: 2 ran, 0 skipped, 0 failed'});

    it('counts pages in the progress and keeps the Run button active', async () => {
        const p = page({statuses: [status(), done]});
        await p.choose('Run whole collection');
        await p.context.startRun();
        expect(p.els.runProgressText.textContent).toBe('Running… (1 of 2 pages done)');
        expect(p.els.runToolbarBtn.classList.contains('active')).toBe(true);
        p.context.closeRunProgress();
        p.menu();
        expect(p.els.runProgressPopover.style.display).toBe('block');
        expect(p.context.buildArrangePopover).toHaveBeenCalledTimes(1);
        await vi.advanceTimersByTimeAsync(1000);
        expect(p.els.runProgressText.textContent).toBe('Run finished: 2 pages: 2 ran, 0 skipped, 0 failed. See run-log.txt.');
        expect(p.els.runToolbarBtn.classList.contains('active')).toBe(false);
    });

    it('locks the open page while it is in the run and leaves other pages editable', async () => {
        const p = page({statuses: [status(), status(), status()]});
        const bottom = p.context.panelControllers.bottom;
        await p.choose('Run whole collection');
        await p.context.startRun();
        p.context.selectedYamlPath = '/data/p2/results.yaml';
        await vi.advanceTimersByTimeAsync(1000);
        expect(bottom.setTextRunLocked).toHaveBeenLastCalledWith(true);
        expect(p.els.runPausedNotice.hidden).toBe(false);
        p.context.selectedYamlPath = '/data/p3/results.yaml'; // opened on its own, not from the list
        p.context.currentFolderIndex = -1;
        await vi.advanceTimersByTimeAsync(1000);
        expect(bottom.setTextRunLocked).toHaveBeenLastCalledWith(false);
        expect(bottom.setImageRunLocked).toHaveBeenLastCalledWith(false);
        expect(p.els.runPausedNotice.hidden).toBe(true);
    });

    it('counts a page of the list as in the run even when the backend spells its folder differently', async () => {
        const spelt = {folders: ['/private/data/p1', '/private/data/p2']};
        const p = page({statuses: [status(spelt), {...done, ...spelt}]});
        await p.choose('Run whole collection');
        await p.context.startRun();
        p.context.selectedYamlPath = '/data/p2/results.yaml';
        p.context.currentFolderIndex = 1;
        expect(p.context.runLocksOpenPage()).toBe(true);
        await vi.advanceTimersByTimeAsync(1000);
        expect(p.context.loadCollectraFolderInner).toHaveBeenCalledWith(1);
    });

    it('reloads the open page when it finishes only if the page was in the run', async () => {
        const inRun = page({statuses: [status(), done]});
        await inRun.choose('Run whole collection');
        await inRun.context.startRun();
        inRun.context.selectedYamlPath = '/data/p2/results.yaml';
        await vi.advanceTimersByTimeAsync(1000);
        expect(inRun.context.loadCollectraFolderInner).toHaveBeenCalled();
        const outside = page({statuses: [status(), done]});
        await outside.choose('Run whole collection');
        await outside.context.startRun();
        outside.context.selectedYamlPath = '/data/p3/results.yaml';
        outside.context.currentFolderIndex = -1;
        await vi.advanceTimersByTimeAsync(1000);
        expect(outside.context.loadCollectraFolderInner).not.toHaveBeenCalled();
    });
});

it('a single-image run still locks only its own page, even from the list', async () => {
    const one = {running: true, finished: false, folder: '/private/data/p1', current: null, done: 0, total: 1,
        summary: null, error: null};
    const p = page({statuses: [one]});
    await p.choose('Run this image');
    await p.context.startRun();
    p.context.selectedYamlPath = '/data/p2/results.yaml';
    p.context.currentFolderIndex = 1;
    expect(p.context.runLocksOpenPage()).toBe(false);
});

it('the Run button closes its menu when the menu is open', () => {
    const p = page();
    p.menu();
    p.context.showRunMenu({currentTarget: p.els.runToolbarBtn});
    expect(p.els.nodeContextMenu).toBeUndefined();
    expect(p.context.buildArrangePopover).toHaveBeenCalledTimes(1);
});

it('a press on the Run button does not close its menu before the click toggles it', () => {
    const at = html.indexOf('function onNodeContextMenuOutsideClick');
    const handler = html.slice(at, html.indexOf('\n        }', at) + 10);
    const button = {contains: (target) => target === 'run'};
    const menu = {dataset: {owner: 'run'}, contains: () => false};
    const context = {closeNodeContextMenu: vi.fn(),
        document: {getElementById: (id) => ({nodeContextMenu: menu, runToolbarBtn: button})[id]}};
    vm.createContext(context);
    vm.runInContext(handler, context);
    context.onNodeContextMenuOutsideClick({target: 'run'});
    expect(context.closeNodeContextMenu).not.toHaveBeenCalled();
    context.onNodeContextMenuOutsideClick({target: 'elsewhere'});
    expect(context.closeNodeContextMenu).toHaveBeenCalled();
});

it('showRunDialog unticks the tick itself', async () => {
    const p = page();
    p.els.runOverwriteInput.checked = true;
    await p.context.showRunDialog(true);
    expect(p.els.runOverwriteInput.checked).toBe(false);
});

describe('while the tick fetches the preview again', () => {
    const deferred = () => {
        let resolve;
        const promise = new Promise((done) => { resolve = done; });
        return {promise, resolve};
    };
    const preview = (name) => collectionPreview({tasks: [{name, pages: 2, replace: 0}]});

    it('shows only the latest response, even when an older one arrives last', async () => {
        const p = page();
        await p.choose('Run whole collection');
        const slow = deferred();
        const fast = deferred();
        p.api.get_collection_preview.mockReturnValueOnce(slow.promise).mockReturnValueOnce(fast.promise);
        const first = p.context.toggleRunOverwrite(true);
        const second = p.context.toggleRunOverwrite(false);
        fast.resolve(preview('latest'));
        await second;
        slow.resolve(preview('older'));
        await first;
        expect(p.shown()[0]).toBe('latest — runs on 2 pages');
        expect(p.els.runModalSummary.textContent).toMatch(/Nothing existing is replaced\.$/);
    });

    it('disables Run until the latest response is shown, so the start sends the visible tick', async () => {
        const p = page({statuses: [{running: false, finished: true, folder: '', folders: [], unit: 'pages',
            current: null, done: 0, total: 0, summary: '0 pages: 0 ran', error: null}]});
        await p.choose('Run whole collection');
        const pending = deferred();
        p.api.get_collection_preview.mockReturnValueOnce(pending.promise);
        const toggled = p.context.toggleRunOverwrite(true);
        expect(p.els.runModalConfirm.disabled).toBe(true);
        expect(p.els.runOverwriteInput.disabled).toBe(true); // no second preview while the backend plans
        await p.context.startRun();
        expect(p.api.start_collection_run).not.toHaveBeenCalled();
        pending.resolve(preview('ocr'));
        await toggled;
        expect(p.els.runModalConfirm.disabled).toBe(false);
        expect(p.els.runOverwriteInput.disabled).toBe(false);
        await p.context.startRun();
        expect(p.api.start_collection_run).toHaveBeenCalledWith('', true);
    });
});
