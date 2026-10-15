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
        window: {
            pywebview: {api: {activate_layout: vi.fn().mockResolvedValue(result)}},
            location: {reload: vi.fn()},
        },
    };
    vm.createContext(context);
    vm.runInContext(html.slice(start, end), context);
    return context;
}

it('selects a preset with the active page and pauses autosave before rebuilding', async () => {
    const ctx = setup({success: true});
    await ctx.selectSavedLayout('custom:Review.yaml');
    expect(ctx.window.pywebview.api.activate_layout).toHaveBeenCalledWith('custom:Review.yaml', 4);
    expect(ctx.clearTimeout).toHaveBeenCalledWith(123);
    expect(ctx.arrangementReady).toBe(false);
    expect(ctx.window.location.reload).toHaveBeenCalledOnce();
});

it('keeps the viewer open and restores autosave when a layout cannot load', async () => {
    const ctx = setup({success: false, error: 'Missing layout'});
    await ctx.selectSavedLayout('pipeline:0');
    expect(ctx.window.location.reload).not.toHaveBeenCalled();
    expect(ctx.arrangementReady).toBe(true);
    expect(ctx.alert).toHaveBeenCalledWith('Could not load layout: Missing layout');
});
