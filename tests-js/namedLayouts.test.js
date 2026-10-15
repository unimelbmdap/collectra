const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync(require.resolve('../collectra/gui/index.html'), 'utf8');
const start = html.indexOf('        async function selectSavedLayout(id)');
const end = html.indexOf('        function showSaveLayoutDialog()', start);

function setup(result) {
    const context = {
        arrangementReady: true,
        arrangementSaveTimer: 123,
        currentFolderIndex: 4,
        clearTimeout: vi.fn(),
        alert: vi.fn(),
        closeNodeContextMenu: vi.fn(),
        restoreArrangement: vi.fn().mockResolvedValue(),
        scheduleArrangementSave: vi.fn(),
        window: {
            pywebview: {api: {load_layout: vi.fn().mockResolvedValue(result)}},
            location: {reload: vi.fn()},
        },
    };
    vm.createContext(context);
    vm.runInContext(html.slice(start, end), context);
    return context;
}

it('applies only the panels in place without reloading the page or results', async () => {
    const arrangement = {version: 1, panels: {}};
    const ctx = setup({success: true, arrangement});
    await ctx.selectSavedLayout('custom:Review.yaml');
    expect(ctx.window.pywebview.api.load_layout).toHaveBeenCalledWith('custom:Review.yaml');
    expect(ctx.clearTimeout).toHaveBeenCalledWith(123);
    expect(ctx.arrangementReady).toBe(false);
    expect(ctx.restoreArrangement).toHaveBeenCalledWith(arrangement, true);
    expect(ctx.scheduleArrangementSave).toHaveBeenCalledOnce();
    expect(ctx.window.location.reload).not.toHaveBeenCalled();
});

it('keeps the viewer open and restores autosave when a layout cannot load', async () => {
    const ctx = setup({success: false, error: 'Missing layout'});
    await ctx.selectSavedLayout('pipeline:0');
    expect(ctx.window.location.reload).not.toHaveBeenCalled();
    expect(ctx.arrangementReady).toBe(true);
    expect(ctx.alert).toHaveBeenCalledWith('Could not load layout: Missing layout');
});

it('restores closed original panels and retains dynamic views for reuse', () => {
    const top = {rootEl: {style: {flex: 'old'}}};
    const bottom = {rootEl: {style: {flex: 'old'}}};
    const dynamic = {rootEl: {}, destroyDagTab: vi.fn()};
    const registry = {bottom: {}, panel2: {}};
    const ctx = {
        originalLayoutPanels: {top, bottom},
        originalCollapseEntries: {top: {}, bottom: {}},
        panelControllers: {bottom, panel2: dynamic},
        panelCollapseRegistry: registry,
        panelLayoutContainer: {replaceChildren: vi.fn()},
        originalPanelDivider: {},
        reusableLayoutPanels: null,
        nextDynamicPanelIndex: 10,
        focusedPanel: 'panel2',
        cy: {}, lastDagNodes: [],
    };
    vm.createContext(ctx);
    const begin = html.indexOf('        function resetPanelsForLayout()');
    const finish = html.indexOf('        function releaseUnusedLayoutPanels()', begin);
    vm.runInContext(html.slice(begin, finish), ctx);
    ctx.resetPanelsForLayout();
    expect(ctx.panelControllers).toEqual({top, bottom});
    expect(ctx.panelLayoutContainer.replaceChildren).toHaveBeenCalledWith(top.rootEl, ctx.originalPanelDivider, bottom.rootEl);
    expect(ctx.reusableLayoutPanels.get('panel2').controller).toBe(dynamic);
    expect(dynamic.destroyDagTab).not.toHaveBeenCalled();
    expect(top.rootEl.style.flex).toBe('');
    expect(ctx.nextDynamicPanelIndex).toBe(1);
});
