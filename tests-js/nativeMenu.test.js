const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(require.resolve('../collectra/gui/nativeMenu.js'), 'utf8');

function setup() {
    const clicked = {};
    const input = {focus: vi.fn(), select: vi.fn()};
    const controller = {
        panel: {getActiveTab: () => ({id: 'tab1'})}, closeTab: vi.fn(),
        pageViewer: {viewport: {goHome: vi.fn()}},
    };
    const ctx = {
        apiReady: true, focusedPanel: 'bottom', panelControllers: {bottom: controller},
        document: {
            querySelector: () => null,
            getElementById: (id) => id === 'folderSearchInput' ? input : {click: () => {clicked[id] = true;}},
        },
        collectraFolders: [{yaml_path: '/page/results.yaml'}], currentFolderIndex: 0,
        selectedYamlPath: '/page/results.yaml',
        sidebarRegion: {collapsed: true}, panelCollapseRegistry: {},
        window: {pywebview: {api: {reveal_in_folder: vi.fn().mockResolvedValue({success: true})}}},
        console, alert: vi.fn(),
    };
    for (const name of ['selectCollectra', 'goPrevFolder', 'goNextFolder', 'toggleSidebarCollapse',
        'openArtefactSearch', 'stepArtefactSearch', 'openDagTab', 'resetDag', 'createPanel',
        'isOnlyExpandedPanel', 'updatePanelCloseButtons', 'removePanelByKey', 'showSaveLayoutDialog',
        'openStatsModal', 'openInfoModal']) ctx[name] = vi.fn();
    ctx.showUpdatesDialog = vi.fn();
    vm.createContext(ctx);
    vm.runInContext(source, ctx);
    return {ctx, controller, clicked, input};
}

it('uses existing navigation and split actions without changing results itself', async () => {
    const {ctx} = setup();
    await ctx.runNativeMenuAction('next-file');
    await ctx.runNativeMenuAction('previous-file');
    await ctx.runNativeMenuAction('split-right');
    expect(ctx.goNextFolder).toHaveBeenCalledOnce();
    expect(ctx.goPrevFolder).toHaveBeenCalledOnce();
    expect(ctx.createPanel).toHaveBeenCalledWith('right');
});

it('opens the live layout chooser and save dialog', async () => {
    const {ctx, clicked} = setup();
    await ctx.runNativeMenuAction('choose-layout');
    await ctx.runNativeMenuAction('save-layout');
    expect(clicked.newPanelToolbarBtn).toBe(true);
    expect(ctx.showSaveLayoutDialog).toHaveBeenCalledOnce();
});

it('opens Check for Updates from the native menu', async () => {
    const {ctx} = setup();
    await ctx.runNativeMenuAction('check-updates');
    expect(ctx.showUpdatesDialog).toHaveBeenCalledOnce();
});

it('focuses file search and expands its sidebar', async () => {
    const {ctx, input} = setup();
    await ctx.runNativeMenuAction('find-file');
    expect(ctx.toggleSidebarCollapse).toHaveBeenCalledOnce();
    expect(input.focus).toHaveBeenCalledOnce();
    expect(input.select).toHaveBeenCalledOnce();
});

it('targets the focused panel and current results file', async () => {
    const {ctx, controller} = setup();
    await ctx.runNativeMenuAction('close-tab');
    await ctx.runNativeMenuAction('fit-image');
    await ctx.runNativeMenuAction('reveal');
    expect(controller.closeTab).toHaveBeenCalledWith('tab1');
    expect(controller.pageViewer.viewport.goHome).toHaveBeenCalledOnce();
    expect(ctx.window.pywebview.api.reveal_in_folder).toHaveBeenCalledWith('/page/results.yaml');
});

it('leaves dialogs and startup alone', async () => {
    const {ctx} = setup();
    ctx.apiReady = false;
    await ctx.runNativeMenuAction('open');
    ctx.apiReady = true;
    ctx.document.querySelector = () => ({});
    await ctx.runNativeMenuAction('open');
    expect(ctx.selectCollectra).not.toHaveBeenCalled();
});
