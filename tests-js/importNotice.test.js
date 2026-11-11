const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync(require.resolve('../collectra/gui/index.html'), 'utf8');
const {importNoticeText} = require('../collectra/gui/importNotice.js');

const notice = (extra = {}) => ({summary: 'Imported 2 pages', imported: 2, existing: 0, skipped: [], failed: [], ...extra});

describe('importNoticeText', () => {
    it('is empty when nothing was imported', () => {
        expect(importNoticeText(undefined)).toBe('');
        expect(importNoticeText({})).toBe('');
    });

    it('is the summary when everything went well', () => {
        expect(importNoticeText(notice())).toBe('Imported 2 pages');
    });

    it('leaves skipped files to the summary', () => {
        const info = notice({summary: 'Imported 2 pages, skipped 1 file Grapto doesn\'t accept', skipped: ['notes.txt: not accepted']});
        expect(importNoticeText(info)).toBe('Imported 2 pages, skipped 1 file Grapto doesn\'t accept');
    });

    it('adds one line for each failed file', () => {
        const info = notice({summary: 'Imported 1 page, failed 2', failed: ['a.jpg: read-only', 'b.png: no space']});
        expect(importNoticeText(info)).toBe('Imported 1 page, failed 2\nFailed: a.jpg: read-only\nFailed: b.png: no space');
    });
});

describe('the page', () => {
    it('loads the notice script next to the run dialog script', () => {
        expect(html).toContain('<script src="runDialog.js"></script>\n    <script src="importNotice.js"></script>');
    });
});

// ── Behaviour: the page's own functions against stand-in elements ──

const fn = (name, kind = 'function') => {
    const from = html.indexOf(`${kind === 'async' ? 'async ' : ''}function ${name}(`);
    return html.slice(from, html.indexOf('\n        }\n', from)) + '\n        }\n';
};
const ready = html.slice(html.indexOf("window.addEventListener('pywebviewready'"), html.lastIndexOf('</script>'));

function element() {
    const attributes = {};
    return {
        textContent: '', className: '', title: '', disabled: false,
        setAttribute(name, value) { attributes[name] = String(value); },
        getAttribute(name) { return attributes[name]; },
    };
}

function page(api) {
    const els = {selectCollectraBtn: element(), loadResult: element()};
    els.selectCollectraBtn.setAttribute('aria-label', 'Load');
    const context = {
        importNoticeText, apiReady: true, console,
        document: {getElementById: (id) => els[id]},
        window: {pywebview: {api}, addEventListener: (name, handler) => { context.ready = handler; }},
        handleSingleFolderResult: vi.fn().mockResolvedValue(undefined),
        handleParentFolderResult: vi.fn().mockResolvedValue(undefined),
        showPipelineWithoutResults: vi.fn().mockResolvedValue(undefined),
        loadWorkflowAssets: vi.fn().mockResolvedValue(undefined),
        restoreArrangement: vi.fn().mockResolvedValue(undefined),
        startupActions: (initial) => ({handleParentFolderResult: Boolean(initial.provided)}),
        openInfoModal: vi.fn(),
    };
    vm.createContext(context);
    vm.runInContext(fn('showResult') + fn('showImportNotice') + fn('selectCollectra', 'async') + ready, context);
    return {els, context};
}

const parent = (extra = {}) => ({success: true, mode: 'parent', folders: [{name: 'p1'}], ...extra});

describe('opening a folder', () => {
    it('shows what the import did, in the load details', async () => {
        const {els, context} = page({select_collectra_path: async () => parent({import: notice()})});
        await context.selectCollectra();
        expect(context.handleParentFolderResult).toHaveBeenCalled();
        expect(els.loadResult.textContent).toBe('Imported 2 pages');
        expect(els.loadResult.className).toBe('result success');
        expect(context.openInfoModal).toHaveBeenCalled();
    });

    it('shows a failed import as an error', async () => {
        const result = {success: true, mode: 'single', import: notice({summary: 'Failed 1', failed: ['a.jpg: read-only']})};
        const {els, context} = page({select_collectra_path: async () => result});
        await context.selectCollectra();
        expect(context.handleSingleFolderResult).toHaveBeenCalled();
        expect(els.loadResult.textContent).toBe('Failed 1\nFailed: a.jpg: read-only');
        expect(els.loadResult.className).toBe('result error');
    });

    it('shows nothing extra when nothing was imported', async () => {
        const {els, context} = page({select_collectra_path: async () => parent()});
        await context.selectCollectra();
        expect(els.loadResult.textContent).toBe('');
        expect(context.openInfoModal).not.toHaveBeenCalled();
    });

    it('shows names from the folder as text', async () => {
        const info = notice({summary: 'Failed 1', failed: ['<b>x</b>.jpg: read-only']});
        const {els, context} = page({select_collectra_path: async () => parent({import: info})});
        await context.selectCollectra();
        expect(els.loadResult.textContent).toBe('Failed 1\nFailed: <b>x</b>.jpg: read-only');
        expect(els.loadResult.innerHTML).toBeUndefined();
    });
});

describe('the load button while a folder opens', () => {
    const label = (els) => els.selectCollectraBtn.getAttribute('aria-label');

    async function whilePending(outcome) {
        let finish;
        const {els, context} = page({select_collectra_path: () => new Promise((resolve) => { finish = resolve; })});
        const opening = context.selectCollectra();
        expect(label(els)).toBe('Opening…');
        expect(els.selectCollectraBtn.title).toBe('Opening…');
        expect(els.selectCollectraBtn.disabled).toBe(true);
        finish(outcome);
        await opening;
        return els;
    }

    it.each([
        ['a folder opened', parent()],
        ['an error', {success: false, error: 'No results.yaml found in folder'}],
        ['a cancelled dialog', {success: false, error: 'No folder selected'}],
    ])('says Opening… and is back to Load after %s', async (_, outcome) => {
        const els = await whilePending(outcome);
        expect(label(els)).toBe('Load');
        expect(els.selectCollectraBtn.title).toBe('');
        expect(els.selectCollectraBtn.disabled).toBe(false);
    });

    it('is back to Load when the call throws', async () => {
        const {els, context} = page({select_collectra_path: async () => { throw new Error('bridge gone'); }});
        await context.selectCollectra();
        expect(label(els)).toBe('Load');
        expect(els.selectCollectraBtn.disabled).toBe(false);
        expect(els.loadResult.className).toBe('result error');
    });
});

describe('start-up', () => {
    it('shows what the import of the command line inputs did', async () => {
        const initial = {success: true, provided: true, mode: 'parent', folders: [{name: 'p1'}], import: notice()};
        const {els, context} = page({get_initial_items: async () => initial});
        await context.ready();
        expect(context.handleParentFolderResult).toHaveBeenCalledWith(initial);
        expect(els.loadResult.textContent).toBe('Imported 2 pages');
        expect(context.openInfoModal).toHaveBeenCalled();
    });

    it('shows nothing extra without an import', async () => {
        const {els, context} = page({get_initial_items: async () => ({success: true, provided: false})});
        await context.ready();
        expect(context.showPipelineWithoutResults).toHaveBeenCalled();
        expect(els.loadResult.textContent).toBe('');
        expect(context.openInfoModal).not.toHaveBeenCalled();
    });
});
