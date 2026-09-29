const {displayArtefact, registerArtefactRenderer} = require('../collectra/gui/artefactDisplay.js');

function panel() {
    return {
        clearPageArtefact: vi.fn(),
        renderPageArtefact: vi.fn(),
        renderTextNodeAsPage: vi.fn(),
        el: {textDisplay: {style: {}}, contentDiv: {style: {}, textContent: ''}},
    };
}

it('routes custom image types and crops to the same OpenSeadragon view', async () => {
    for (const type of ['custom.SpectralTile', 'collectra.ImageCrop', 'collectra.Link']) {
        const controller = panel();
        const view = {kind: 'image', source: 'data:image/png;base64,abc', target_id: 'crop', annotations: []};
        await displayArtefact(controller, {id: 'link', type, view});
        expect(controller.renderPageArtefact).toHaveBeenCalledWith(undefined, view);
    }
});

it.each(['plain', 'markdown', 'xml'])('uses the declared %s format regardless of node name', async (format) => {
    const controller = panel();
    const view = {kind: 'text', format, text: '', editable: true};
    await displayArtefact(controller, {id: 'arbitrary-name', type: 'custom.Notes', view});
    expect(controller.clearPageArtefact).toHaveBeenCalled();
    expect(controller.renderTextNodeAsPage).toHaveBeenCalledWith('arbitrary-name', 'custom.Notes', {...view, target_id: 'arbitrary-name'});
});

it('renders properties as text without interpreting HTML', async () => {
    const controller = panel();
    await displayArtefact(controller, {id: 'custom', view: {kind: 'properties', data: {value: '<script>bad()</script>'}}});
    expect(controller.el.contentDiv.textContent).toContain('<script>bad()</script>');
});

it('supports registering an additional renderer', async () => {
    const renderer = vi.fn();
    registerArtefactRenderer('spectrum', renderer);
    const controller = panel();
    await displayArtefact(controller, {id: 'sample', view: {kind: 'spectrum', values: [1, 2]}});
    expect(renderer).toHaveBeenCalledWith(controller, {kind: 'spectrum', values: [1, 2], target_id: 'sample'}, expect.anything());
});
