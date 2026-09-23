// Views are selected by Artefact.display(), independent of Python type names.
// Renderers receive the panel controller and a JSON-compatible view descriptor.
const artefactRenderers = new Map([
    ['image', async (controller, view) => controller.renderPageArtefact(view.image_path, view)],
    ['text', async (controller, view, result) => {
        controller.clearPageArtefact();
        await controller.renderTextNodeAsPage(result.id, result.type, view);
    }],
    ['properties', async (controller, view) => {
        controller.clearPageArtefact();
        if (controller.el.textDisplay) controller.el.textDisplay.style.display = 'none';
        controller.el.contentDiv.style.display = 'block';
        controller.el.contentDiv.textContent = JSON.stringify(view.data, null, 2);
    }],
]);

function registerArtefactRenderer(kind, renderer) {
    artefactRenderers.set(kind, renderer);
}

async function displayArtefact(controller, result) {
    const view = {...result.view};
    view.target_id = view.target_id || result.id;
    const renderer = artefactRenderers.get(view.kind);
    if (!renderer) throw new Error(`Unknown artefact view: ${view.kind}`);
    await renderer(controller, view, result);
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {displayArtefact, registerArtefactRenderer};
}
