const {resizeMaskBrush, mountSegmentationEditor} = require('../collectra/gui/displays/segmentationEditor.js');

function element() {
    return {
        style: {}, children: [], handlers: {}, attrs: {},
        append(...children) {this.children.push(...children);},
        appendChild(child) {this.children.push(child);},
        setAttribute(key, value) {this.attrs[key] = value;},
        addEventListener(key, value) {this.handlers[key] = value;},
        removeEventListener(key) {delete this.handlers[key];},
        getBoundingClientRect: () => ({left: 10, top: 20}),
        setPointerCapture() {}, hasPointerCapture: () => false, releasePointerCapture() {}, remove() {},
    };
}

async function setup(save) {
    const commands = [];
    const context = {};
    for (const name of ['clearRect', 'fillRect', 'drawImage', 'beginPath', 'moveTo', 'lineTo', 'stroke', 'arc', 'fill']) {
        context[name] = (...args) => commands.push({name, args, composite: context.globalCompositeOperation, fillStyle: context.fillStyle});
    }
    vi.stubGlobal('document', {...element(), createElement: () => Object.assign(element(), {getContext: () => context}),
        createTextNode: text => ({textContent: text}), querySelector: () => null});
    vi.stubGlobal('Image', class {decode = async () => {};});
    vi.stubGlobal('getComputedStyle', () => ({getPropertyValue: () => ''}));
    vi.stubGlobal('OpenSeadragon', {Button: function(options) {
        this.element = element(); this.element.onclick = options.onClick; this.options = options;
        this.imgRest = {src: options.srcRest}; this.imgGroup = {}; this.imgHover = {}; this.imgDown = {};
        this.destroy = vi.fn();
    }, Point: class {constructor(x,y) {this.x = x; this.y = y;}},
        ControlAnchor: {BOTTOM_LEFT: 1, TOP_LEFT: 2}});
    const segmentation = {mask_source: 'mask', width: 100, height: 80, orientation: 'north', bounds: {x: 10, y: 10, w: 20, h: 20}};
    const api = vi.fn(save || (async () => ({success: true, changed: true, segmentation})));
    vi.stubGlobal('window', {...element(), pywebview: {api: {update_segmentation_mask: api}}});
    const surface = element();
    const controls = [];
    const viewer = {
        canvas: surface, container: surface, gestureSettingsMouse: {},
        buttonGroup: {buttons: []}, addButton(button) {this.buttonGroup.buttons.push(button);},
        world: {getItemAt: () => ({imageToViewportRectangle: (...args) => args,
            viewportToImageCoordinates: p => ({x: p.x * 2, y: p.y * 2}),
            imageToViewportCoordinates: (x, y) => ({x,y})})},
        viewport: {pointFromPixel: p => p, pixelFromPoint: p => ({x: p.x / 2, y: p.y / 2})},
        addOverlay: vi.fn(), removeOverlay: vi.fn(),
        addControl: control => controls.push(control), removeControl: vi.fn(),
        addHandler: vi.fn(), removeHandler: vi.fn(),
    };
    const controller = {_artefactToken: {}, pageViewer: viewer, pageAnno: {setDrawingEnabled: vi.fn()},
        syncDrawBoxButton: vi.fn(), zoomToBoxGeometry: vi.fn()};
    const onSaved = vi.fn();
    const editor = await mountSegmentationEditor(controller, {target_id: 'line', segmentation},
        {token: controller._artefactToken, resultsPath: '/original/results.yaml', onSaved});
    controller._maskEditor = editor;
    const [label, status] = controls[0].children;
    const [brushButton, eraserButton] = viewer.buttonGroup.buttons;
    const brush = brushButton.element, eraser = eraserButton.element;
    const event = (extra = {}) => ({button: 0, pointerId: 7, clientX: 30, clientY: 35,
        preventDefault: vi.fn(), stopImmediatePropagation: vi.fn(), ...extra});
    return {editor, api, surface, commands, brush, eraser, brushButton, eraserButton, document, window, input: label.children[0], status, controller, onSaved, event};
}

afterEach(() => vi.unstubAllGlobals());

it('dims the whole image with 50% black and removes that shade inside the mask', async () => {
    const {commands, controller} = await setup();
    expect(commands.find(c => c.name === 'fillRect')).toMatchObject({args: [0,0,100,80], fillStyle: 'rgba(0,0,0,0.5)'});
    expect(commands.find(c => c.name === 'drawImage').composite).toBe('destination-out');
    expect(controller.zoomToBoxGeometry).toHaveBeenCalledWith({x: 10, y: 10, w: 20, h: 20});
});

it.each(['brush', 'eraser'])('paints circular %s strokes in image coordinates and autosaves them to the original page', async tool => {
    const s = await setup();
    s[tool].onclick();
    expect(s.editor.isActive()).toBe(true);
    expect(s.controller.pageViewer.gestureSettingsMouse.dragToPan).toBe(false);
    s.surface.handlers.pointerdown(s.event());
    s.surface.handlers.pointermove(s.event({clientX: 35}));
    s.surface.handlers.pointerup(s.event());
    await vi.waitFor(() => expect(s.status.textContent).toBe('Saved'));
    expect(s.api).toHaveBeenCalledWith('line', [{tool, radius: 10, points: [[40,30], [50,30]]}], '/original/results.yaml', 'north');
    expect(s.commands.some(c => c.name === 'arc' && c.args[2] === 10)).toBe(true);
    expect(s.onSaved).toHaveBeenCalled();
    s[tool].onclick();
    expect(s.editor.isActive()).toBe(false);
    expect(s.controller.pageViewer.gestureSettingsMouse.dragToPan).toBe(true);
});

it('uses Shift + wheel to resize while leaving normal zoom gestures alone', async () => {
    const s = await setup(); s.brush.onclick();
    const zoom = s.event({deltaY: -10});
    s.surface.handlers.wheel(zoom);
    expect(zoom.preventDefault).not.toHaveBeenCalled();
    const resize = s.event({shiftKey: true, deltaY: -10});
    s.surface.handlers.wheel(resize);
    expect(resize.preventDefault).toHaveBeenCalled();
    expect(Number(s.input.value)).toBeGreaterThan(20);
    expect(resizeMaskBrush(4096, -1)).toBe(4096);
    expect(resizeMaskBrush(0.5, 1)).toBe(0.5);
});

it('saves an unfinished stroke when navigating away without updating the next view', async () => {
    let finish;
    const s = await setup(() => new Promise(resolve => {finish = resolve;}));
    s.brush.onclick(); s.surface.handlers.pointerdown(s.event());
    s.editor.dispose();
    expect(s.api).toHaveBeenCalled();
    expect(s.surface.handlers.pointerdown).toBeUndefined();
    finish({success: true, segmentation: {mask_source: 'updated'}});
    await Promise.resolve(); await Promise.resolve();
    expect(s.onSaved).not.toHaveBeenCalled();
});

it('reports save errors and restores the last saved mask', async () => {
    const s = await setup(async () => ({success: false, error: 'Disk full'}));
    s.eraser.onclick(); s.surface.handlers.pointerdown(s.event()); s.surface.handlers.pointerup(s.event());
    await vi.waitFor(() => expect(s.status.textContent).toBe('Disk full'));
    expect(s.commands.filter(c => c.name === 'drawImage')).toHaveLength(2);
    expect(s.onSaved).not.toHaveBeenCalled();
});

 it('uses matching native icon buttons in the viewer navigation group with an active state', async () => {
    const s = await setup();
    const normal = s.brushButton.imgRest.src;
    expect(normal).toMatch(/^data:image\/svg\+xml,/);
    expect(decodeURIComponent(normal)).toContain('width="35" height="34"');
    expect(s.brush.attrs['aria-label']).toBe('Brush');
    expect(s.eraser.attrs['aria-label']).toBe('Eraser');
    s.brush.onclick();
    expect(s.brush.attrs['aria-pressed']).toBe('true');
    expect(s.brushButton.imgRest.src).not.toBe(normal);
    s.eraser.onclick();
    expect(s.brush.attrs['aria-pressed']).toBe('false');
    expect(s.eraser.attrs['aria-pressed']).toBe('true');
    s.editor.dispose();
    expect(s.controller.pageViewer.buttonGroup.buttons).toHaveLength(0);
    expect(s.brushButton.destroy).toHaveBeenCalled();
});

it('temporarily uses Shift for brush and Option for eraser, restoring the selected tool', async () => {
    const s = await setup();
    s.surface.handlers.pointerenter(s.event());
    s.document.handlers.keydown({key: 'Shift', shiftKey: true});
    expect(s.brush.attrs['aria-pressed']).toBe('true');
    s.document.handlers.keydown({key: 'Alt', shiftKey: true, altKey: true});
    expect(s.eraser.attrs['aria-pressed']).toBe('true');
    s.document.handlers.keyup({key: 'Alt', shiftKey: true});
    expect(s.brush.attrs['aria-pressed']).toBe('true');
    s.document.handlers.keyup({key: 'Shift'});
    expect(s.editor.isActive()).toBe(false);
    s.brush.onclick();
    s.document.handlers.keydown({key: 'Alt', altKey: true});
    expect(s.eraser.attrs['aria-pressed']).toBe('true');
    s.document.handlers.keyup({key: 'Alt'});
    expect(s.brush.attrs['aria-pressed']).toBe('true');
});

it.each([['Shift', 'shiftKey', 'brush'], ['Alt', 'altKey', 'eraser']])(
    'paints with held %s without clicking a tool button', async (key, modifier, tool) => {
        const s = await setup();
        s.document.handlers.keydown({key, [modifier]: true});
        s.surface.handlers.pointerenter(s.event({[modifier]: true}));
        s.surface.handlers.pointerdown(s.event({[modifier]: true}));
        s.surface.handlers.pointerup(s.event({[modifier]: true}));
        await vi.waitFor(() => expect(s.status.textContent).toBe('Saved'));
        expect(s.api.mock.calls[0][1][0].tool).toBe(tool);
        s.document.handlers.keyup({key});
        expect(s.editor.isActive()).toBe(false);
    }
);

it('ignores modifier shortcuts away from the image or while editing a field', async () => {
    const s = await setup();
    s.document.handlers.keydown({key: 'Shift', shiftKey: true});
    expect(s.editor.isActive()).toBe(false);
    s.surface.handlers.pointerenter(s.event());
    s.document.activeElement = {closest: () => ({})};
    s.document.handlers.keydown({key: 'Alt', altKey: true});
    expect(s.editor.isActive()).toBe(false);
});

it('restores the prior mode on pointer leave and window blur and removes keyboard listeners', async () => {
    const s = await setup();
    s.surface.handlers.pointerenter(s.event({shiftKey: true}));
    expect(s.editor.isActive()).toBe(true);
    s.surface.handlers.pointerleave();
    expect(s.editor.isActive()).toBe(false);
    s.document.handlers.keydown({key: 'Alt', altKey: true});
    s.surface.handlers.pointerenter(s.event({altKey: true}));
    expect(s.eraser.attrs['aria-pressed']).toBe('true');
    s.window.handlers.blur();
    expect(s.editor.isActive()).toBe(false);
    s.editor.dispose();
    expect(s.document.handlers.keydown).toBeUndefined();
    expect(s.document.handlers.keyup).toBeUndefined();
    expect(s.window.handlers.blur).toBeUndefined();
});

it('uses a hollow red cursor for erasing, keeps the button icon, and no longer activates on Control', async () => {
    const s = await setup();
    const icon = s.eraserButton.options.srcRest;
    s.surface.handlers.pointerenter(s.event());
    s.document.handlers.keydown({key: 'Control', ctrlKey: true});
    expect(s.editor.isActive()).toBe(false);
    s.document.handlers.keydown({key: 'Alt', altKey: true});
    expect(s.surface.children[0].className).toContain('segmentation-eraser-cursor');
    expect(s.eraserButton.options.srcRest).toBe(icon);
    s.document.handlers.keyup({key: 'Alt'});
    expect(s.surface.children[0].className).not.toContain('segmentation-eraser-cursor');
    const css = require('node:fs').readFileSync(require.resolve('../collectra/gui/displays/segmentationEditor.css'), 'utf8');
    expect(css).toContain('border: 2px solid #ff3b30');
    expect(css).toContain('background: transparent');
});
