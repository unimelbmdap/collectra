// While the open page is being run, its editors cannot change it.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {mountSegmentationEditor} = require('../collectra/gui/displays/segmentationEditor.js');

const gui = path.resolve(__dirname, '../collectra/gui');
const read = (file) => fs.readFileSync(path.join(gui, file), 'utf8');
const between = (text, start, end) => {
    const from = text.indexOf(start);
    return text.slice(from, text.indexOf(end, from));
};

afterEach(() => vi.unstubAllGlobals());

// ── Text editors ──

const textSource = between(read('displays/text.js'),
    '        PanelController.prototype.renderTextNodeAsPage =', '        // The slot at the left');

function textEditor({locked, editable = true} = {}) {
    const made = [];
    function Controller() {}
    const ctx = {
        PanelController: Controller,
        createTextAutosave: () => ({changed() {}, dispose() {}, flush() {}}),
        EasyMDE: function(options) {
            const classes = new Set();
            this.options = options;
            this.value = () => options.initialValue;
            this.toggleSideBySide = vi.fn();
            this.gui = {toolbar: {classList: {
                toggle: (name, on) => (on ? classes.add(name) : classes.delete(name)),
                contains: (name) => classes.has(name),
            }}};
            this.codemirror = {setOption: vi.fn(), refresh() {}, on() {}, off() {}};
            made.push(this);
        },
        window: {pywebview: {api: {update_display_text: vi.fn()}}},
        selectedYamlPath: '/p1/results.yaml',
        requestAnimationFrame() {}, markSearchInEditor() {}, flashStatus() {}, rerunArtefactSearch() {}, console,
    };
    if (locked !== undefined) ctx.runLocksOpenPage = () => locked;
    vm.createContext(ctx);
    vm.runInContext(textSource, ctx);
    const controller = new Controller();
    Object.assign(controller, {
        el: {textDisplay: {style: {}, querySelector: () => ({})}, headerId: {}},
        stopTextAutosave() {}, clearNodeTools() {}, moveEditorToolbar() {}, mountTextZoom() {}, renderPager() {},
    });
    const view = {format: 'markdown', editable, text: 'hello'};
    return {controller, made, view, ctx};
}

it('opens the text of the running page read-only, with its formatting tools disabled', async () => {
    const t = textEditor({locked: true});
    await t.controller.renderTextNodeAsPage('words', null, t.view);
    const editor = t.made[0];
    expect(editor.options.readOnly).toBe(true);
    expect(editor.gui.toolbar.classList.contains('disabled-for-preview')).toBe(true);
});

it('makes the text editable again when the run ends', async () => {
    const t = textEditor({locked: true});
    await t.controller.renderTextNodeAsPage('words', null, t.view);
    const editor = t.made[0];
    t.controller.setTextRunLocked(false);
    expect(editor.codemirror.setOption).toHaveBeenLastCalledWith('readOnly', false);
    expect(editor.gui.toolbar.classList.contains('disabled-for-preview')).toBe(false);
    t.controller.setTextRunLocked(true);
    expect(editor.codemirror.setOption).toHaveBeenLastCalledWith('readOnly', true);
    expect(editor.gui.toolbar.classList.contains('disabled-for-preview')).toBe(true);
});

it('leaves the text of another page editable', async () => {
    const t = textEditor({locked: false});
    await t.controller.renderTextNodeAsPage('words', null, t.view);
    expect(t.made[0].options.readOnly).toBe(false);
    expect(t.made[0].gui.toolbar.classList.contains('disabled-for-preview')).toBe(false);
});

it('opens text editable when no run code is loaded', async () => {
    const t = textEditor();
    await t.controller.renderTextNodeAsPage('words', null, t.view);
    expect(t.made[0].options.readOnly).toBe(false);
});

it('keeps text that cannot be edited read-only after the run ends', async () => {
    const t = textEditor({locked: true, editable: false});
    await t.controller.renderTextNodeAsPage('words', null, t.view);
    t.controller.setTextRunLocked(false);
    expect(t.made[0].codemirror.setOption).toHaveBeenLastCalledWith('readOnly', true);
});

it('does nothing to text when no editor is open', () => {
    const t = textEditor();
    expect(() => t.controller.setTextRunLocked(true)).not.toThrow();
});

// ── Boxes on the page image ──

const imageSource = read('displays/image.js');
const boxSource = between(imageSource,
    '        PanelController.prototype.syncDrawBoxButton =', '        PanelController.prototype.showPageImage =');

function boxEditor(locked) {
    function Controller() {}
    const ctx = {
        PanelController: Controller,
        resolveDagLabel: () => null,
        cropTypeOptionsForController: () => ({labels: ['crop']}),
        drawBoxImages: () => ({armed: {}, normal: {}}),
    };
    if (locked !== undefined) ctx.runLocksOpenPage = () => locked;
    vm.createContext(ctx);
    vm.runInContext(boxSource, ctx);
    const controller = new Controller();
    let drawing = true;
    Object.assign(controller, {
        displayView: {can_create_crop: true},
        pageViewer: {isFullPage: () => false, gestureSettingsMouse: {}},
        pageAnno: {
            isDrawingEnabled: () => drawing,
            setDrawingEnabled: vi.fn((on) => { drawing = on; }),
            setUserSelectAction: vi.fn(),
        },
        drawBoxBtn: {
            element: {disabled: false},
            disable: vi.fn(function() { this.element.disabled = true; }),
            enable: vi.fn(function() { this.element.disabled = false; }),
        },
        el: {imageDisplay: {classList: {toggle() {}}}},
    });
    return {controller, ctx};
}

it('stops boxes on the running page from being drawn, moved or resized', () => {
    const {controller} = boxEditor(true);
    controller.setImageRunLocked(true);
    expect(controller.pageAnno.setDrawingEnabled).toHaveBeenCalledWith(false);
    expect(controller.pageAnno.setUserSelectAction).toHaveBeenLastCalledWith('SELECT');
    expect(controller.drawBoxBtn.element.disabled).toBe(true);
});

it('lets boxes be edited again when the run ends', () => {
    const b = boxEditor(true);
    b.controller.setImageRunLocked(true);
    b.ctx.runLocksOpenPage = () => false;
    b.controller.setImageRunLocked(false);
    expect(b.controller.pageAnno.setUserSelectAction).toHaveBeenLastCalledWith('EDIT');
    expect(b.controller.drawBoxBtn.element.disabled).toBe(false);
});

it('leaves boxes on another page editable', () => {
    const {controller} = boxEditor(false);
    controller.syncDrawBoxButton();
    expect(controller.drawBoxBtn.element.disabled).toBe(false);
    expect(boxEditor().controller.syncDrawBoxButton()).toBeUndefined();
});

it('does nothing to boxes when no image is open', () => {
    const {controller} = boxEditor(true);
    controller.pageAnno = null;
    controller.pageViewer = null;
    expect(() => controller.setImageRunLocked(true)).not.toThrow();
});

it('opens the annotator of the running page with box editing off', () => {
    const created = imageSource.indexOf('this.pageAnno = AnnotoriousOSD.createOSDAnnotator(');
    const next = imageSource.slice(created, imageSource.indexOf('\n\n', created));
    expect(next).toContain("if (typeof runLocksOpenPage === 'function' && runLocksOpenPage()) this.pageAnno.setUserSelectAction('SELECT');");
});

// ── Segmentation masks ──

function surfaceElement() {
    return {
        style: {}, children: [], handlers: {}, attrs: {},
        append(...children) { this.children.push(...children); },
        appendChild(child) { this.children.push(child); },
        setAttribute(key, value) { this.attrs[key] = value; },
        addEventListener(key, value) { this.handlers[key] = value; },
        removeEventListener(key) { delete this.handlers[key]; },
        getBoundingClientRect: () => ({left: 10, top: 20}),
        setPointerCapture() {}, hasPointerCapture: () => false, releasePointerCapture() {}, remove() {},
    };
}

async function maskEditor(locked) {
    const context = {};
    for (const name of ['clearRect', 'fillRect', 'drawImage', 'beginPath', 'moveTo', 'lineTo', 'stroke', 'arc', 'fill',
        'save', 'restore', 'scale']) {
        context[name] = () => {};
    }
    vi.stubGlobal('document', {...surfaceElement(), createElement: () => Object.assign(surfaceElement(), {getContext: () => context}),
        createTextNode: (text) => ({textContent: text}), querySelector: () => null});
    vi.stubGlobal('Image', class { decode = async () => {}; });
    vi.stubGlobal('getComputedStyle', () => ({getPropertyValue: () => ''}));
    vi.stubGlobal('segmentationBoundsInPreview', () => null);
    vi.stubGlobal('runLocksOpenPage', () => locked);
    vi.stubGlobal('OpenSeadragon', {Button: function(options) {
        this.element = surfaceElement(); this.element.onclick = options.onClick;
        this.imgRest = {}; this.imgGroup = {}; this.imgHover = {}; this.imgDown = {}; this.destroy = vi.fn();
    }, Point: class { constructor(x, y) { this.x = x; this.y = y; } },
    ControlAnchor: {BOTTOM_LEFT: 1, TOP_LEFT: 2}});
    const segmentation = {mask_source: 'mask', width: 100, height: 80, orientation: 'north'};
    const save = vi.fn(async () => ({success: true, segmentation}));
    vi.stubGlobal('window', {...surfaceElement(), pywebview: {api: {update_segmentation_mask: save}}});
    const surface = surfaceElement();
    const viewer = {
        canvas: surface, container: surface, gestureSettingsMouse: {},
        buttonGroup: {buttons: []}, addButton(button) { this.buttonGroup.buttons.push(button); },
        world: {getItemAt: () => ({getContentSize: () => ({x: 50, y: 40}), imageToViewportRectangle: (...args) => args,
            viewportToImageCoordinates: (p) => ({x: p.x * 2, y: p.y * 2}), imageToViewportCoordinates: (x, y) => ({x, y})})},
        viewport: {pointFromPixel: (p) => p, pixelFromPoint: (p) => ({x: p.x / 2, y: p.y / 2})},
        addOverlay() {}, removeOverlay() {}, addControl() {}, removeControl() {}, addHandler() {}, removeHandler() {},
    };
    const controller = {_artefactToken: {}, pageViewer: viewer, pageAnno: {setDrawingEnabled() {}},
        syncDrawBoxButton() {}, zoomToBoxGeometry() {}};
    const editor = await mountSegmentationEditor(controller, {target_id: 'line', segmentation},
        {token: controller._artefactToken, resultsPath: '/p1/results.yaml'});
    const brush = viewer.buttonGroup.buttons[0].element;
    const event = {button: 0, pointerId: 7, clientX: 30, clientY: 35, preventDefault() {}, stopImmediatePropagation() {}};
    return {editor, save, surface, brush, event};
}

it.each([[true, 0], [false, 1]])('paints a mask stroke only when the page is not being run (locked: %s)', async (locked, saves) => {
    const m = await maskEditor(locked);
    m.brush.onclick();
    m.surface.handlers.pointerdown(m.event);
    m.surface.handlers.pointerup(m.event);
    await Promise.resolve();
    expect(m.save).toHaveBeenCalledTimes(saves);
});
