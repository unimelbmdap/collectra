const {createTextAutosave} = require('../collectra/gui/textAutosave.js');

afterEach(() => vi.useRealTimers());

function setup() {
    vi.useFakeTimers();
    let value = 'original';
    const write = vi.fn().mockResolvedValue({success: true});
    const report = vi.fn();
    const save = createTextAutosave({initialValue: value, read: () => value, write, report});
    return {save, write, report, edit: (next) => {value = next; save.changed();}};
}

it('debounces changes and skips unchanged content', async () => {
    const {save, edit, write} = setup();
    edit('first');
    await vi.advanceTimersByTimeAsync(250);
    edit('latest');
    await vi.advanceTimersByTimeAsync(500);
    expect(write.mock.calls).toEqual([['latest']]);
    await save.flush();
    expect(write).toHaveBeenCalledTimes(1);
});

it('flushes pending changes immediately when leaving an editor', async () => {
    const {save, edit, write} = setup();
    edit('last edit');
    await save.dispose();
    expect(write).toHaveBeenCalledWith('last edit');
    await vi.runAllTimersAsync();
    expect(write).toHaveBeenCalledTimes(1);
});

it('serializes in-flight saves and persists the newest edit last', async () => {
    const {save, edit, write} = setup();
    let finish;
    write.mockImplementationOnce(() => new Promise((resolve) => {finish = resolve;}));
    edit('first');
    const pending = save.flush();
    edit('latest');
    save.flush();
    expect(write).toHaveBeenCalledTimes(1);
    finish({success: true});
    await pending;
    expect(write.mock.calls).toEqual([['first'], ['latest']]);
});

it('reports failures and retries unsaved content on manual save', async () => {
    const {save, edit, write, report} = setup();
    write.mockResolvedValueOnce({success: false, error: 'Disk full'});
    edit('new text');
    await save.flush();
    expect(report).toHaveBeenCalledWith('Error: Disk full', true);
    await save.flush();
    expect(write).toHaveBeenCalledTimes(2);
    expect(report).toHaveBeenLastCalledWith('Saved', false);
});

it('does not write when edits are reverted before the debounce', async () => {
    const {edit, write} = setup();
    edit('temporary');
    edit('original');
    await vi.runAllTimersAsync();
    expect(write).not.toHaveBeenCalled();
});
