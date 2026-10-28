// A start-up that finds its layout different from the usual (e.g. the original top DAG panel was
// closed in an earlier session and restored closed) must still open the page.
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync(require.resolve('../collectra/gui/index.html'), 'utf8');
const fn = (name, kind = 'function') => {
    const from = html.indexOf(`${kind === 'async' ? 'async ' : ''}function ${name}(`);
    return html.slice(from, html.indexOf('\n        }\n', from)) + '\n        }\n';
};

const graph = { success: true, ext: 'grapto', nodes: [{ id: 'a', nodeType: 'data' }], edges: [] };

function dagPage({ withElements, initThrows }) {
    const els = {};
    const element = (id) => ({ id, disabled: true, style: {}, dataset: {}, textContent: '' });
    const context = {
        apiReady: true, lastDagNodes: [], lastDagEdges: [], panelControllers: {}, console: { error() {} },
        document: { getElementById: (id) => (withElements || id.endsWith('ToolbarBtn') ? (els[id] = els[id] || element(id)) : null) },
        window: { pywebview: { api: { get_pipeline_graph: async () => graph } } },
        createDagInstance: vi.fn(), populateTabs: vi.fn(), setActive: vi.fn(), activeNodeIds: null, cy: null,
    };
    vm.createContext(context);
    vm.runInContext(fn('initCytoscape') + fn('loadAndRenderDag', 'async'), context);
    if (initThrows) context.initCytoscape = () => { throw new Error('could not draw'); };
    return { context, els };
}

describe('drawing the DAG when its panel is gone', () => {
    it('initCytoscape does nothing, quietly, when there is no DAG canvas', () => {
        const { context } = dagPage({ withElements: false });
        expect(() => context.initCytoscape([], [])).not.toThrow();
        expect(context.createDagInstance).not.toHaveBeenCalled();
    });

    it('loadAndRenderDag finishes without error, and the lists are still enabled', async () => {
        const { context, els } = dagPage({ withElements: false });
        await expect(context.loadAndRenderDag()).resolves.toBeUndefined();
        expect(context.lastDagNodes).toEqual(graph.nodes);
        expect(els.artefactsToolbarBtn.disabled).toBe(false);
    });

    it('a failure while drawing does not escape, with or without the placeholder to report it on', async () => {
        for (const withElements of [false, true]) {
            const { context } = dagPage({ withElements, initThrows: true });
            await expect(context.loadAndRenderDag()).resolves.toBeUndefined();
        }
    });

    it('reports the failure on the placeholder when there is one', async () => {
        const { context, els } = dagPage({ withElements: true, initThrows: true });
        await context.loadAndRenderDag();
        expect(els.dagPlaceholder.textContent).toContain('could not draw');
    });
});

describe('opening the page at start-up', () => {
    const source = fn('handleParentFolderResult', 'async');

    function startup(dagBehaviour) {
        const loaded = [];
        const element = () => ({ innerHTML: '', textContent: '', style: {}, value: '' });
        const context = {
            workflowAssetsLoaded: true, collectraFolders: [], currentFolderIndex: 0, console: { error() {} },
            populateSidebar: vi.fn(), clearCurrentView: vi.fn(), updateNavButtons: vi.fn(), displayGlobalStats: vi.fn(),
            clearDag: vi.fn(), refreshTables: vi.fn(), loadWorkflowAssets: async () => {},
            loadAndRenderDag: dagBehaviour,
            loadCollectraFolder: async (index) => { loaded.push(index); },
            document: { getElementById: () => element() },
        };
        vm.createContext(context);
        vm.runInContext(source, context);
        return { loaded, run: (result) => context.handleParentFolderResult(result) };
    }
    const result = { folders: [{ name: 'a' }, { name: 'b' }, { name: 'c' }], active_index: 2, ext: 'grapto' };

    it('opens the remembered page', async () => {
        const { loaded, run } = startup(async () => {});
        await run(result);
        expect(loaded).toEqual([2]);
    });

    it('opens page 1 when none is remembered', async () => {
        const { loaded, run } = startup(async () => {});
        await run({ ...result, active_index: undefined });
        expect(loaded).toEqual([0]);
    });

    it('still opens the page when drawing the DAG fails', async () => {
        const { loaded, run } = startup(async () => { throw new Error('no canvas'); });
        await run(result);
        expect(loaded).toEqual([2]);
    });
});

it('loads and opens the pipeline graph without loading result data', async () => {
    const panel = {openDagKindTab: vi.fn()};
    const context = {loadAndRenderDag: vi.fn(), openingPanel: () => panel};
    vm.createContext(context);
    vm.runInContext(fn('showPipelineWithoutResults', 'async'), context);
    await context.showPipelineWithoutResults();
    expect(context.loadAndRenderDag).toHaveBeenCalledOnce();
    expect(panel.openDagKindTab).toHaveBeenCalledOnce();
});
