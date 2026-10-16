// Safe fallback for artefacts that have no specialized view.
async function renderPropertiesArtefact(controller, view) {
    controller.clearPageArtefact();
    if (controller.el.placeholder) controller.el.placeholder.style.display = 'none';
    if (controller.el.textDisplay) controller.el.textDisplay.style.display = 'none';
    controller.el.contentDiv.style.display = 'block';
    controller.el.contentDiv.textContent = JSON.stringify(view.data, null, 2);
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {renderPropertiesArtefact};
} else {
    registerArtefactRenderer('properties', renderPropertiesArtefact);
}
