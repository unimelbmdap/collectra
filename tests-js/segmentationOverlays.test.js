const fs = require('node:fs');
const vm = require('node:vm');
const {nodeColor} = require('../collectra/gui/nodeColors.js');
const html = fs.readFileSync(require.resolve('../collectra/gui/index.html'), 'utf8');
const start = html.indexOf('PanelController.prototype.refreshPageAnnotations =');
const source = html.slice(start, html.indexOf('\n        };', start) + 11);

function viewer(decode = async () => {}) {
    const paint = {drawImage: vi.fn(), fillRect: vi.fn()};
    const canvas = {style: {}, getContext: () => paint};
    const context = {
        PanelController: function() {}, nodeColor, calculateGeometryFromCropRegion: () => null,
        Image: class { naturalWidth = 10; naturalHeight = 5; decode = decode; },
        document: {createElement: () => canvas},
    };
    vm.createContext(context);
    vm.runInContext(source, context);
    const controller = new context.PanelController();
    Object.assign(controller, {
        _artefactToken: {}, displayView: {},
        pageAnno: {clearAnnotations: vi.fn()},
        pageViewer: {
            world: {_contentSize: {x: 200, y: 100}, getItemAt: () => ({imageToViewportRectangle: (...args) => args})},
            addOverlay: vi.fn(), removeOverlay: vi.fn(),
        },
    });
    return {controller, canvas, paint};
}

const row = {id: 'line1', label: 'line_image', mask_source: 'data:image/png;base64,test',
    mask_region: {x: 0.1, y: 0.2, width: 0.3, height: 0.4}};

it('tints the actual mask using the DAG label colour and positions it in image coordinates', async () => {
    const {controller, canvas, paint} = viewer();
    await controller.refreshPageAnnotations(null, [row]);
    expect(paint.globalCompositeOperation).toBe('source-in');
    expect(paint.fillStyle).toBe(nodeColor('line_image'));
    expect(canvas.style.opacity).toBe('0.35');
    expect(controller.pageViewer.addOverlay).toHaveBeenCalledWith({element: canvas, location: [20, 20, 60, 40], checkResize: false});
    await controller.refreshPageAnnotations(null, []);
    expect(controller.pageViewer.removeOverlay).toHaveBeenCalledWith(canvas);
});

it('does not add a stale mask after navigation during image decoding', async () => {
    let finish;
    const {controller} = viewer(() => new Promise(resolve => { finish = resolve; }));
    const pending = controller.refreshPageAnnotations(null, [row]);
    controller._artefactToken = {};
    finish();
    await pending;
    expect(controller.pageViewer.addOverlay).not.toHaveBeenCalled();
});
