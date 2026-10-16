// Dispatch only: built-in implementations live in displays/, while artefact
// packages may provide a self-contained ES module in view.renderer.
const artefactRenderers = new Map();
const rendererModules = new Map();

function registerArtefactRenderer(kind, renderer) {
    if (typeof kind !== 'string' || !kind || typeof renderer !== 'function') {
        throw new TypeError('A renderer needs a view kind and a render function');
    }
    artefactRenderers.set(kind, renderer);
}

function disposeArtefactRenderer(controller) {
    controller._customDisplayAbort?.abort();
    controller._customDisplayAbort = null;
    const dispose = controller._customDisplayDispose;
    controller._customDisplayDispose = null;
    if (dispose) {
        try { dispose(); } catch (error) { console.error('Artefact renderer cleanup failed:', error); }
    }
}

async function loadArtefactRenderer(source) {
    if (!source.startsWith('data:text/javascript;base64,')) {
        throw new Error('Custom renderers must be published with DisplayContext.renderer()');
    }
    if (!rendererModules.has(source)) {
        rendererModules.set(source, import(/* @vite-ignore */ source).then(module => {
            if (typeof module.default !== 'function') throw new Error('An artefact renderer must export a default function');
            return module.default;
        }).catch(error => {
            rendererModules.delete(source);
            throw error;
        }));
    }
    return rendererModules.get(source);
}

async function renderCustomArtefact(controller, view, result) {
    controller.clearPageArtefact();
    // clearPageArtefact also disposes the previous custom renderer.
    const abort = controller._customDisplayAbort = new AbortController();
    const host = controller.el.contentDiv;
    if (controller.el.placeholder) controller.el.placeholder.style.display = 'none';
    if (controller.el.textDisplay) controller.el.textDisplay.style.display = 'none';
    host.style.display = 'block';
    const element = document.createElement('div');
    element.className = 'artefact-custom-display';
    host.replaceChildren(element);
    const renderer = await loadArtefactRenderer(view.renderer);
    if (abort.signal.aborted) return;
    const dispose = await renderer({
        element, view, identity: {id: result.id, type: result.type},
        api: window.pywebview.api, signal: abort.signal,
    });
    if (dispose !== undefined && typeof dispose !== 'function') {
        throw new Error('An artefact renderer must return a cleanup function or undefined');
    }
    if (abort.signal.aborted) {
        dispose?.();
    } else {
        controller._customDisplayDispose = dispose;
    }
}

async function displayArtefact(controller, result) {
    const view = {...result.view};
    view.target_id = view.target_id || result.id;
    const renderer = view.renderer ? renderCustomArtefact : artefactRenderers.get(view.kind);
    if (!renderer) throw new Error(`Unknown artefact view: ${view.kind}`);
    await renderer(controller, view, result);
}

if (typeof module !== 'undefined' && module.exports) {
    // CommonJS test/embedding entry point; the browser loads each script explicitly.
    registerArtefactRenderer('image', require('./displays/image.js').renderImageArtefact);
    registerArtefactRenderer('text', require('./displays/text.js').renderTextArtefact);
    registerArtefactRenderer('properties', require('./displays/properties.js').renderPropertiesArtefact);
    module.exports = {displayArtefact, registerArtefactRenderer, disposeArtefactRenderer};
}
