const {displayArtefact, disposeArtefactRenderer} = require('../collectra/gui/artefactDisplay.js');

function panel() {
    const controller = {
        el: {contentDiv: {style: {}, replaceChildren(child) { this.child = child; }},
            textDisplay: {style: {}}, placeholder: {style: {}}},
        clearPageArtefact() { disposeArtefactRenderer(this); },
    };
    return controller;
}

const moduleUrl = source => 'data:text/javascript;base64,' + Buffer.from(source).toString('base64');
const result = source => ({id: 'example', type: 'plugin.Measurement',
    view: {kind: 'measurement', value: 42, renderer: moduleUrl(source)}});

beforeEach(() => {
    vi.stubGlobal('document', {createElement: () => ({})});
    vi.stubGlobal('window', {pywebview: {api: {}}});
});
afterEach(() => vi.unstubAllGlobals());

it('loads a standalone module, passes its view, and disposes it when leaving', async () => {
    const controller = panel();
    await displayArtefact(controller, result(`export default function({element, view, identity, signal}) {
        element.textContent = identity.id + ':' + view.value;
        element.signal = signal;
        return () => { element.disposed = true; };
    }`));
    const mounted = controller.el.contentDiv.child;
    expect(mounted.textContent).toBe('example:42');
    expect(mounted.signal.aborted).toBe(false);
    controller.clearPageArtefact();
    expect(mounted.signal.aborted).toBe(true);
    expect(mounted.disposed).toBe(true);
});

it('cleans up an asynchronous renderer after navigation without replacing the new view', async () => {
    const controller = panel();
    let release, started;
    const ready = new Promise(resolve => { started = resolve; });
    window.pywebview.api.wait = () => { started(); return new Promise(resolve => {release = resolve;}); };
    const pending = displayArtefact(controller, result(`export default async function({element, api}) {
        await api.wait();
        element.textContent = 'old view';
        return () => { element.disposed = true; };
    }`));
    await ready;
    const previous = controller.el.contentDiv.child;
    controller.clearPageArtefact();
    controller.el.contentDiv.replaceChildren({textContent: 'new view'});
    release();
    await pending;
    expect(previous.disposed).toBe(true);
    expect(controller.el.contentDiv.child.textContent).toBe('new view');
});

it('reports renderer load errors and invalid modules', async () => {
    await expect(displayArtefact(panel(), result('export const value = 1;'))).rejects.toThrow('default function');
    const invalid = result('');
    invalid.view.renderer = 'file:///arbitrary.js';
    await expect(displayArtefact(panel(), invalid)).rejects.toThrow('DisplayContext.renderer');
});

it('reports errors thrown by a renderer', async () => {
    await expect(displayArtefact(panel(), result('export default function() { throw new Error("broken view"); }'))).rejects.toThrow('broken view');
});
