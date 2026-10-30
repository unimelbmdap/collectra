const {setupSourceControl} = require('../collectra/gui/sourceControl.js');

function element(dataset = {}) {
    const listeners = {};
    const classes = new Set();
    return {
        dataset, value: '', textContent: '', disabled: false,
        classList: {add: name => classes.add(name), remove: name => classes.delete(name), contains: name => classes.has(name)},
        addEventListener: (name, handler) => { listeners[name] = handler; },
        emit: async name => { await listeners[name]?.({}); await new Promise(resolve => setTimeout(resolve, 0)); },
        focus: vi.fn(), setAttribute: vi.fn(),
    };
}

function setup() {
    const ids = ['gitModal', 'gitCommitModal', 'gitOutput', 'gitRepository', 'gitCommitMessage',
        'gitCommitConfirm', 'gitCommitError', 'gitToolbarBtn', 'gitClose', 'gitCommitCancel', 'gitPullStrategy'];
    const nodes = Object.fromEntries(ids.map(id => [id, element()]));
    const buttons = Object.fromEntries(['status', 'pull', 'add', 'commit', 'push'].map(action => [action, element({gitAction: action})]));
    const strategies = Object.fromEntries(['merge', 'rebase', 'ff-only'].map(name => [name, element({pullStrategy: name})]));
    nodes.gitModal.querySelectorAll = selector => Object.values(selector.includes('pull-strategy') ? strategies : buttons);
    nodes.gitPullStrategy.hidden = true;
    const document = {getElementById: id => nodes[id], addEventListener: vi.fn(), activeElement: element()};
    const api = {git_action: vi.fn().mockResolvedValue({success: true, repository: '/repo', output: 'Ready'})};
    setupSourceControl(document, () => api);
    return {nodes, buttons, strategies, api};
}

it('opens with status and pins subsequent operations to the displayed repository', async () => {
    const {nodes, buttons, api} = setup();
    await nodes.gitToolbarBtn.emit('click');
    expect(nodes.gitModal.classList.contains('visible')).toBe(true);
    expect(api.git_action).toHaveBeenCalledWith('status', '', '');
    expect(nodes.gitRepository.textContent).toBe('/repo');
    for (const action of ['pull', 'add', 'push']) {
        await buttons[action].emit('click');
        expect(api.git_action).toHaveBeenLastCalledWith(action, '', '/repo');
    }
});

it('asks for a message, shows commit errors, and closes the popup on success', async () => {
    const {nodes, buttons, api} = setup();
    await nodes.gitToolbarBtn.emit('click');
    await buttons.commit.emit('click');
    expect(api.git_action).toHaveBeenCalledTimes(1);
    expect(nodes.gitCommitModal.classList.contains('visible')).toBe(true);
    expect(nodes.gitCommitConfirm.disabled).toBe(true);
    nodes.gitCommitMessage.value = 'A message';
    await nodes.gitCommitMessage.emit('input');
    expect(nodes.gitCommitConfirm.disabled).toBe(false);
    api.git_action.mockResolvedValueOnce({success: false, error: 'Nothing staged'});
    await nodes.gitCommitConfirm.emit('click');
    expect(nodes.gitCommitError.textContent).toBe('Nothing staged');
    expect(nodes.gitCommitModal.classList.contains('visible')).toBe(true);
    await nodes.gitCommitConfirm.emit('click');
    expect(api.git_action).toHaveBeenLastCalledWith('commit', 'A message', '/repo');
    expect(nodes.gitCommitModal.classList.contains('visible')).toBe(false);
});

it('blocks duplicate actions while Git is running and displays output as text', async () => {
    const {nodes, buttons, api} = setup();
    let complete;
    api.git_action.mockReturnValueOnce(new Promise(resolve => {complete = resolve;}));
    await nodes.gitToolbarBtn.emit('click');
    expect(buttons.add.disabled).toBe(true);
    await buttons.add.emit('click');
    expect(api.git_action).toHaveBeenCalledTimes(1);
    complete({success: true, repository: '/repo', output: '<script>literal output</script>'});
    await new Promise(resolve => setTimeout(resolve, 0));
    expect(nodes.gitOutput.textContent).toBe('<script>literal output</script>');
    expect(buttons.add.disabled).toBe(false);
});

it('offers merge, rebase and fast-forward only when pull needs a strategy, then retries with the choice', async () => {
    const {nodes, buttons, strategies, api} = setup();
    await nodes.gitToolbarBtn.emit('click');
    expect(nodes.gitPullStrategy.hidden).toBe(true);
    api.git_action.mockResolvedValueOnce({success: false, needs_strategy: true, error: 'divergent'});
    await buttons.pull.emit('click');
    expect(nodes.gitPullStrategy.hidden).toBe(false);
    expect(nodes.gitOutput.textContent).toContain('diverged');
    await strategies.rebase.emit('click');
    expect(api.git_action).toHaveBeenLastCalledWith('pull', '', '/repo', 'rebase');
    expect(nodes.gitPullStrategy.hidden).toBe(true);
});

it('hides the strategy choice again after any other action', async () => {
    const {nodes, buttons, api} = setup();
    await nodes.gitToolbarBtn.emit('click');
    api.git_action.mockResolvedValueOnce({success: false, needs_strategy: true, error: 'divergent'});
    await buttons.pull.emit('click');
    expect(nodes.gitPullStrategy.hidden).toBe(false);
    await buttons.status.emit('click');
    expect(nodes.gitPullStrategy.hidden).toBe(true);
});
