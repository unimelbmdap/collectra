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
        'gitCommitConfirm', 'gitCommitError', 'gitToolbarBtn', 'gitClose', 'gitCommitCancel'];
    const nodes = Object.fromEntries(ids.map(id => [id, element()]));
    const buttons = Object.fromEntries(['status', 'pull', 'save', 'push'].map(action => [action, element({gitAction: action})]));
    nodes.gitModal.querySelectorAll = () => Object.values(buttons);
    const document = {getElementById: id => nodes[id], addEventListener: vi.fn(), activeElement: element()};
    const api = {git_action: vi.fn().mockResolvedValue({success: true, repository: '/repo', output: 'Ready'})};
    setupSourceControl(document, () => api);
    return {nodes, buttons, api};
}

it('opens with status and pins subsequent operations to the displayed repository', async () => {
    const {nodes, buttons, api} = setup();
    await nodes.gitToolbarBtn.emit('click');
    expect(nodes.gitModal.classList.contains('visible')).toBe(true);
    expect(api.git_action).toHaveBeenCalledWith('status', '', '');
    expect(nodes.gitRepository.textContent).toBe('Project folder: /repo');
    for (const action of ['pull', 'push']) {
        await buttons[action].emit('click');
        expect(api.git_action).toHaveBeenLastCalledWith(action, '', '/repo');
    }
});

it('asks for a message, shows save errors, and closes the popup on success', async () => {
    const {nodes, buttons, api} = setup();
    await nodes.gitToolbarBtn.emit('click');
    await buttons.save.emit('click');
    expect(api.git_action).toHaveBeenCalledTimes(1);
    expect(nodes.gitCommitModal.classList.contains('visible')).toBe(true);
    expect(nodes.gitCommitError.hidden).toBe(true);
    expect(nodes.gitCommitConfirm.disabled).toBe(true);
    nodes.gitCommitMessage.value = 'A message';
    await nodes.gitCommitMessage.emit('input');
    expect(nodes.gitCommitConfirm.disabled).toBe(false);
    api.git_action.mockResolvedValueOnce({success: false, error: 'There are no changes to save.'});
    await nodes.gitCommitConfirm.emit('click');
    expect(nodes.gitCommitError.textContent).toBe('There are no changes to save.');
    expect(nodes.gitCommitError.hidden).toBe(false);
    expect(nodes.gitCommitModal.classList.contains('visible')).toBe(true);
    await nodes.gitCommitConfirm.emit('click');
    expect(api.git_action).toHaveBeenLastCalledWith('save', 'A message', '/repo');
    expect(nodes.gitCommitModal.classList.contains('visible')).toBe(false);
    expect(nodes.gitCommitError.hidden).toBe(true);
});

it('blocks duplicate actions while Git is running and displays output as text', async () => {
    const {nodes, buttons, api} = setup();
    let complete;
    api.git_action.mockReturnValueOnce(new Promise(resolve => {complete = resolve;}));
    await nodes.gitToolbarBtn.emit('click');
    expect(buttons.pull.disabled).toBe(true);
    await buttons.pull.emit('click');
    expect(api.git_action).toHaveBeenCalledTimes(1);
    complete({success: true, repository: '/repo', output: '<script>literal output</script>'});
    await new Promise(resolve => setTimeout(resolve, 0));
    expect(nodes.gitOutput.textContent).toBe('<script>literal output</script>');
    expect(buttons.pull.disabled).toBe(false);
});

it('says it is working in plain words while an action runs', async () => {
    const {nodes, api} = setup();
    api.git_action.mockReturnValueOnce(new Promise(() => {}));
    await nodes.gitToolbarBtn.emit('click');
    expect(nodes.gitOutput.textContent).toBe('Working…');
});

it('says where the project sits, and when it cannot be found', async () => {
    const {nodes, api} = setup();
    let complete;
    api.git_action.mockReturnValueOnce(new Promise(resolve => {complete = resolve;}));
    await nodes.gitToolbarBtn.emit('click');
    expect(nodes.gitRepository.textContent).toBe('Project folder: …');
    complete({success: false});
    await new Promise(resolve => setTimeout(resolve, 0));
    expect(nodes.gitRepository.textContent).toBe('Project folder: not found');
    expect(nodes.gitOutput.textContent).toBe('Something went wrong.');
});
