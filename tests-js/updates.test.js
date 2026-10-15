const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(require.resolve('../collectra/gui/updates.js'), 'utf8');

function setup(check) {
    function element(tag) {
        return {tag, children: [], style: {}, setAttribute() {}, focus() {}, remove() {},
            append(...items) {this.children.push(...items);},
            appendChild(item) {this.children.push(item);},
            replaceChildren() {this.children = [];},
        };
    }
    const results = element('div');
    const close = element('button');
    const body = element('body');
    const api = {
        check_updates: vi.fn().mockResolvedValue(check),
        install_update: vi.fn().mockResolvedValue({success: true, repository: {
            id: 'pipeline', label: 'Pipeline', status: 'updated', can_update: false, message: 'Restart the GUI',
        }}),
        save_arrangement: vi.fn().mockResolvedValue({success: true}),
        restart_gui: vi.fn().mockResolvedValue({success: true}),
    };
    const ctx = {
        document: {body, activeElement: null, createElement: (tag) => {
            const el = element(tag);
            el.querySelector = (selector) => selector === 'button' ? close : results;
            return el;
        }},
        panelControllers: {}, currentArrangement: () => ({version: 1}), window: {pywebview: {api}},
    };
    vm.createContext(ctx);
    vm.runInContext(source, ctx);
    return {ctx, results, api};
}

it('shows both statuses and updates only after the user selects Update', async () => {
    const {ctx, results, api} = setup({success: true, repositories: [
        {id: 'pipeline', label: 'Pipeline', path: '/pipeline', can_update: true, message: '1 new commit'},
        {id: 'collectra', label: 'Collectra', can_update: false, message: 'Up to date'},
    ]});
    await ctx.showUpdatesDialog();
    expect(results.children).toHaveLength(2);
    expect(api.install_update).not.toHaveBeenCalled();
    const update = results.children[0].children.find((item) => item.tag === 'button');
    expect(update.textContent).toBe('Update Pipeline');
    await update.onclick();
    expect(api.install_update).toHaveBeenCalledWith('pipeline');
    expect(results.children[0].children.some((item) => item.tag === 'button')).toBe(false);
    expect(results.children[0].children[2].textContent).toBe('Restart the GUI');
    const restart = results.children.find((item) => item.textContent === 'Restart Now');
    expect(restart).toBeDefined();
    expect(api.restart_gui).not.toHaveBeenCalled();
    await restart.onclick();
    expect(api.save_arrangement).toHaveBeenCalledWith({version: 1});
    expect(api.restart_gui).toHaveBeenCalledOnce();
});

it('retains the restart prompt when reopening the dialog after an update', async () => {
    const {ctx, results} = setup({success: true, repositories: [], restart_required: true});
    await ctx.showUpdatesDialog();
    expect(results.children.some((item) => item.textContent === 'Restart Now')).toBe(true);
});

it('does not restart if pending edits fail to save', async () => {
    const {ctx, results, api} = setup({success: true, repositories: [], restart_required: true});
    await ctx.showUpdatesDialog();
    ctx.panelControllers.editor = {textAutosave: {flush: vi.fn().mockResolvedValue(false)}};
    await results.children.find((item) => item.textContent === 'Restart Now').onclick();
    expect(api.restart_gui).not.toHaveBeenCalled();
    expect(results.children.some((item) => item.textContent?.includes('Some edits could not be saved'))).toBe(true);
});

it('reports check failures', async () => {
    const {ctx, results} = setup({success: false, error: 'Git unavailable'});
    await ctx.showUpdatesDialog();
    expect(results.textContent).toBe('Could not check for updates: Git unavailable');
});

it('shows blocked repositories without an Update button', async () => {
    const {ctx, results} = setup({success: true, repositories: [
        {id: 'pipeline', label: 'Pipeline', can_update: false, message: 'Local changes'},
    ]});
    await ctx.showUpdatesDialog();
    expect(results.children[0].children.some((item) => item.tag === 'button')).toBe(false);
});
