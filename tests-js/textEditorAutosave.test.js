const vm = require('node:vm');
const {createTextAutosave} = require('../collectra/gui/textAutosave.js');
const html = require('./displaySources.js');
const begin = html.indexOf('        PanelController.prototype.renderTextNodeAsPage =');
const end = html.indexOf('        // The slot at the left', begin);

afterEach(() => vi.useRealTimers());

it.each(['plain', 'markdown', 'xml'])('autosaves %s editor changes to the captured page', async (format) => {
    vi.useFakeTimers();
    const listeners = {};
    let value = 'initial';
    const update = vi.fn().mockResolvedValue({success: true, changed: true});
    function Controller() {}
    const ctx = {
        PanelController: Controller, createTextAutosave,
        EasyMDE: function() {
            this.value = () => value;
            this.toggleSideBySide = vi.fn();
            this.codemirror = {
                setOption() {}, refresh() {},
                on: (event, callback) => {listeners[event] = callback;},
                off: (event) => {delete listeners[event];},
            };
        },
        window: {pywebview: {api: {update_display_text: update}}},
        selectedYamlPath: '/original/results.yaml',
        requestAnimationFrame() {}, markSearchInEditor() {},
        flashStatus() {}, rerunArtefactSearch: vi.fn(), console,
    };
    vm.createContext(ctx);
    vm.runInContext(html.slice(begin, end), ctx);
    const controller = new Controller();
    Object.assign(controller, {
        el: {textDisplay: {style: {}, querySelector: () => ({})}, headerId: {}},
        stopTextAutosave() {}, clearNodeTools() {}, moveEditorToolbar() {}, mountTextZoom() {}, renderPager() {},
    });
    await controller.renderTextNodeAsPage('text', null, {format, editable: true, text: 'initial'});
    expect(update).not.toHaveBeenCalled();
    value = 'edited content';
    listeners.change();
    ctx.selectedYamlPath = '/other/results.yaml';
    await vi.advanceTimersByTimeAsync(500);
    expect(update).toHaveBeenCalledWith('text', 'edited content', '/original/results.yaml');
    await controller.textSave.onSave();
    expect(update).toHaveBeenCalledTimes(1);
    expect(ctx.rerunArtefactSearch).toHaveBeenCalledOnce();
});
