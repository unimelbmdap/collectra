const fs = require('node:fs');
const vm = require('node:vm');
const {nodeColor} = require('../collectra/gui/nodeColors.js');
const html = fs.readFileSync(require.resolve('../collectra/gui/displays/image.js'), 'utf8');
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
    vm.runInContext(fs.readFileSync(require.resolve('../collectra/gui/displays/imageSegmentation.js'), 'utf8') + source, context);
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

function hoverViewer(rows) {
    const handlers = {};
    const tooltip = {style: {}, textContent: ''};
    const context = {
        document: {querySelector: () => null},
        getBoxTooltip: () => tooltip, scheduleHoverHide: vi.fn(), cancelHoverHide: vi.fn(),
        OpenSeadragon: {Point: class {constructor(x, y) {this.x = x; this.y = y;}}},
    };
    vm.createContext(context);
    vm.runInContext(fs.readFileSync(require.resolve('../collectra/gui/displays/imageSegmentation.js'), 'utf8'), context);
    const controller = {
        _artefactToken: {},
        el: {imageDisplay: {
            addEventListener: (name, callback) => {handlers[name] = callback;},
            removeEventListener: (name) => {delete handlers[name];},
        }},
        pageViewer: {
            container: {getBoundingClientRect: () => ({left: 20, top: 30})},
            viewport: {pointFromPixel: p => ({x: p.x * 2, y: p.y * 2})},
            world: {getItemAt: () => ({viewportToImageCoordinates: p => p})},
        },
    };
    const cleanup = context.mountSegmentationHover(controller, rows, {token: controller._artefactToken, imgW: 200, imgH: 100});
    return {context, controller, handlers, tooltip, cleanup};
}

it('shows the artefact name inside its bounding box with zoom and viewer offset applied', () => {
    const {handlers, tooltip, cleanup} = hoverViewer([{...row, name: '<Line one>'}]);
    handlers.pointermove({clientX: 40, clientY: 45});
    expect(tooltip.textContent).toBe('<Line one>');
    expect(tooltip.style.display).toBe('block');
    handlers.pointermove({clientX: 110, clientY: 75});
    expect(tooltip.style.display).toBe('none');
    handlers.pointermove({clientX: 40, clientY: 45});
    cleanup();
    expect(tooltip.style.display).toBe('none');
    expect(handlers.pointermove).toBeUndefined();
});

it('uses the ID if unnamed and prefers the smaller segmentation in overlapping boxes', () => {
    const larger = {...row, id: 'region', name: 'Outer region', mask_region: {x: 0, y: 0, width: 1, height: 1}};
    const {handlers, tooltip} = hoverViewer([larger, row]);
    handlers.pointermove({clientX: 40, clientY: 45});
    expect(tooltip.textContent).toBe('line1');
    handlers.pointermove({clientX: 110, clientY: 75});
    expect(tooltip.textContent).toBe('Outer region');
    handlers.pointerleave();
    expect(tooltip.style.display).toBe('none');
});

it('hides the tooltip over a dialog or after navigation', () => {
    const {context, controller, handlers, tooltip} = hoverViewer([row]);
    handlers.pointermove({clientX: 40, clientY: 45});
    context.document.querySelector = () => ({});
    handlers.pointermove({clientX: 40, clientY: 45});
    expect(tooltip.style.display).toBe('none');
    context.document.querySelector = () => null;
    handlers.pointermove({clientX: 40, clientY: 45});
    controller._artefactToken = {};
    handlers.pointermove({clientX: 40, clientY: 45});
    expect(tooltip.style.display).toBe('none');
});
