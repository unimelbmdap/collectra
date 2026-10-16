// Circular source-pixel brushes with a live 50% black outside-mask overlay.
function resizeMaskBrush(radius, delta) {
    return Math.max(0.5, Math.min(4096, radius * Math.pow(1.15, delta < 0 ? 1 : -1)));
}

function maskToolImages(tool, active) {
    const style = getComputedStyle(document.documentElement);
    const primary = style.getPropertyValue('--app-primary').trim() || '#f09226';
    const dark = style.getPropertyValue('--app-primary-dark').trim() || '#d97f18';
    const svg = (disc, ring, ink) => 'data:image/svg+xml,' + encodeURIComponent(
        '<svg xmlns="http://www.w3.org/2000/svg" width="35" height="34" viewBox="0 0 35 34">' +
        `<circle cx="18" cy="16.5" r="13.5" fill="${disc}" stroke="${ring}"/>` +
        (tool === 'brush'
            ? `<path d="M15 18l-2-2 8-8a2 2 0 0 1 3 3l-8 8z" fill="none" stroke="${ink}" stroke-width="1.7"/><path d="M12 18c-4 0-1 6-5 6 4 2 10-1 8-5z" fill="${ink}"/>`
            : `<path d="M10 20l8-11a2 2 0 0 1 3 0l5 4a2 2 0 0 1 0 3l-7 9h-5zM15 14l9 7M19 25h7" fill="none" stroke="${ink}" stroke-width="1.7" stroke-linejoin="round"/>`) +
        '</svg>');
    return active ? {
        rest: svg(primary, dark, '#fff'), group: svg(primary, dark, '#fff'),
        hover: svg(dark, dark, '#fff'), down: svg(dark, dark, '#fff'),
    } : {
        rest: svg('rgba(244,244,244,0.62)', 'rgba(90,90,90,0.35)', 'rgba(40,40,40,0.62)'),
        group: svg('#ececec', '#b6b6b6', '#333'), hover: svg('#fff', primary, primary),
        down: svg('#e4e4e4', '#8f8f8f', '#222'),
    };
}

async function mountSegmentationEditor(controller, view, {token, resultsPath, onSaved}) {
    if (!view.segmentation) return null;
    const viewer = controller.pageViewer;
    const surface = viewer.canvas;
    const size = view.segmentation;
    const shade = document.createElement('canvas');
    shade.width = size.width;
    shade.height = size.height;
    shade.style.pointerEvents = 'none';
    const context = shade.getContext('2d');
    let saved = size, disposed = false, mode = null, radius = 10, stroke = null, saving = false;
    let maskRequest = 0, maskUpdating = false;
    let lastPointer = null, selectedMode = null, pointerInside = false, shift = false, backtick = false, eraserHeld = false;
    const current = () => !disposed && controller._artefactToken === token;
    async function setMask(data) {
        const request = ++maskRequest;
        maskUpdating = true;
        const image = new Image();
        image.src = data.mask_source;
        try { await image.decode(); }
        finally { if (request === maskRequest) maskUpdating = false; }
        if (!current() || request !== maskRequest) return;
        context.clearRect(0, 0, shade.width, shade.height);
        context.globalCompositeOperation = 'source-over';
        context.fillStyle = 'rgba(0,0,0,0.5)';
        context.fillRect(0, 0, shade.width, shade.height);
        context.globalCompositeOperation = 'destination-out';
        context.drawImage(image, 0, 0);
        context.globalCompositeOperation = 'source-over';
        saved = data;
    }
    await setMask(size);
    if (!current()) return {dispose() {}, isActive: () => false};
    viewer.addOverlay({element: shade, location: viewer.world.getItemAt(0).imageToViewportRectangle(0, 0, size.width, size.height), checkResize: false});

    const bar = document.createElement('div');
    bar.className = 'segmentation-editor-tools';
    bar.setAttribute('role', 'group');
    bar.setAttribute('aria-label', 'Edit segmentation mask');
    const toolButtons = ['brush', 'eraser'].map(tool => {
        const images = maskToolImages(tool, false);
        const name = tool === 'brush' ? 'Brush' : 'Eraser';
        const button = new OpenSeadragon.Button({
            tooltip: name + (tool === 'brush' ? ' — hold Shift' : ' — hold `') + '; click again to pan; Shift + scroll to resize',
            srcRest: images.rest, srcGroup: images.group, srcHover: images.hover, srcDown: images.down,
            onClick: () => choose(tool),
        });
        button.element.setAttribute('role', 'button');
        button.element.setAttribute('aria-label', name);
        button.element.setAttribute('aria-pressed', 'false');
        if (viewer.buttonGroup) viewer.addButton(button);
        else viewer.addControl(button.element, {anchor: OpenSeadragon.ControlAnchor.TOP_LEFT});
        return button;
    });
    const label = document.createElement('label');
    label.textContent = 'Size ';
    const input = document.createElement('input');
    input.type = 'number'; input.min = '1'; input.max = '8192'; input.step = '1'; input.value = '20';
    input.setAttribute('aria-label', 'Brush diameter in image pixels');
    input.title = 'Brush diameter in image pixels. Shift + scroll/zoom to resize.';
    label.append(input, document.createTextNode(' px'));
    const status = document.createElement('span');
    status.setAttribute('role', 'status');
    bar.append(label, status);
    for (const event of ['pointerdown', 'mousedown', 'touchstart', 'click', 'keydown', 'wheel']) {
        bar.addEventListener(event, e => e.stopPropagation());
    }
    viewer.addControl(bar, {anchor: OpenSeadragon.ControlAnchor.BOTTOM_LEFT});
    const ring = document.createElement('div');
    ring.className = 'segmentation-brush-cursor';
    surface.appendChild(ring);
    const previousCursor = surface.style.cursor;
    const updateRing = () => {
        if (!mode || !lastPointer || !current()) { ring.style.display = 'none'; return; }
        const rect = surface.getBoundingClientRect();
        const item = viewer.world.getItemAt(0);
        const a = viewer.viewport.pixelFromPoint(item.imageToViewportCoordinates(0, 0), true);
        const b = viewer.viewport.pixelFromPoint(item.imageToViewportCoordinates(radius * 2, 0), true);
        const diameter = Math.hypot(b.x - a.x, b.y - a.y);
        Object.assign(ring.style, {display: 'block', width: diameter + 'px', height: diameter + 'px',
            left: (lastPointer.clientX - rect.left) + 'px', top: (lastPointer.clientY - rect.top) + 'px'});
    };
    function applyMode() {
        const next = backtick ? 'eraser' : shift ? 'brush' : selectedMode;
        if (mode === next) return;
        if (stroke) finish();
        mode = next;
        ring.className = 'segmentation-brush-cursor' + (mode === 'eraser' ? ' segmentation-eraser-cursor' : '');
        if (mode) controller.pageAnno?.setDrawingEnabled(false);
        toolButtons.forEach((button, index) => {
            const tool = index === 0 ? 'brush' : 'eraser';
            const active = mode === tool;
            const images = maskToolImages(tool, active);
            button.element.setAttribute('aria-pressed', String(active));
            button.imgRest.src = images.rest;
            button.imgGroup.src = images.group;
            button.imgHover.src = images.hover;
            button.imgDown.src = images.down;
        });
        surface.style.cursor = mode ? 'none' : previousCursor;
        controller.syncDrawBoxButton();
        // The editor is assigned to the controller after mounting.
        viewer.gestureSettingsMouse.dragToPan = !mode;
        updateRing();
    }
    function choose(tool) {
        selectedMode = selectedMode === tool ? null : tool;
        applyMode();
    }
    function modifiers(event) {
        const allowed = current() && !document.querySelector('.modal-overlay.visible')
            && !document.activeElement?.closest?.('input, textarea, select, [contenteditable="true"]');
        shift = allowed && !!event.shiftKey;
        backtick = allowed && eraserHeld;
        applyMode();
    }
    const enter = event => { pointerInside = true; modifiers(event); };
    const key = (event, pressed) => {
        const eraserKey = event.key === '`' || event.code === 'Backquote';
        if (eraserKey) eraserHeld = pressed;
        if ((event.key === 'Shift' || eraserKey) && (pointerInside || stroke)) modifiers(event);
    };
    const keydown = event => key(event, true);
    const keyup = event => key(event, false);
    const blur = () => {
        if (stroke) finish();
        pointerInside = false; shift = backtick = eraserHeld = false; lastPointer = null; applyMode(); updateRing();
    };
    input.onchange = () => {
        const diameter = Number(input.value);
        if (Number.isFinite(diameter)) radius = Math.max(0.5, Math.min(4096, diameter / 2));
        input.value = String(Math.round(radius * 2)); updateRing();
    };
    const point = event => {
        const rect = viewer.container.getBoundingClientRect();
        const pixel = new OpenSeadragon.Point(event.clientX - rect.left, event.clientY - rect.top);
        const p = viewer.world.getItemAt(0).viewportToImageCoordinates(viewer.viewport.pointFromPixel(pixel, true));
        return [p.x, p.y];
    };
    const suppress = event => {event.preventDefault(); event.stopImmediatePropagation();};
    function drawSegment(a, b) {
        const path = () => {
            context.beginPath();
            if (a[0] === b[0] && a[1] === b[1]) {
                context.arc(...b, stroke.radius, 0, Math.PI * 2); context.fill();
            } else {
                context.moveTo(...a); context.lineTo(...b);
                context.lineWidth = stroke.radius * 2; context.lineCap = 'round'; context.stroke();
            }
        };
        context.globalCompositeOperation = 'destination-out';
        context.strokeStyle = context.fillStyle = '#000'; path();
        if (stroke.tool === 'eraser') {
            context.globalCompositeOperation = 'source-over';
            context.strokeStyle = context.fillStyle = 'rgba(0,0,0,0.5)'; path();
        }
        context.globalCompositeOperation = 'source-over';
    }
    const down = event => {
        modifiers(event);
        if (!mode || !current() || event.button !== 0
            || document.querySelector('.modal-overlay.visible')) return;
        suppress(event);
        if (saving || maskUpdating) return;
        const p = point(event);
        if (p[0] < 0 || p[1] < 0 || p[0] >= size.width || p[1] >= size.height) return;
        lastPointer = event;
        stroke = {tool: mode, radius, points: [p], pointerId: event.pointerId};
        surface.setPointerCapture(event.pointerId);
        drawSegment(p, p); updateRing();
    };
    const move = event => {
        if (!stroke) pointerInside = true;
        modifiers(event);
        lastPointer = event;
        if (!mode || !current()) return;
        updateRing();
        if (!stroke || event.pointerId !== stroke.pointerId) return;
        suppress(event);
        const p = point(event);
        // Captured drags can leave the source image; clamp at its edge.
        p[0] = Math.max(0, Math.min(size.width - 1, p[0]));
        p[1] = Math.max(0, Math.min(size.height - 1, p[1]));
        drawSegment(stroke.points[stroke.points.length - 1], p);
        stroke.points.push(p);
    };
    async function finish() {
        if (!stroke) return;
        const completed = stroke;
        stroke = null;
        if (surface.hasPointerCapture(completed.pointerId)) surface.releasePointerCapture(completed.pointerId);
        saving = true; status.textContent = 'Saving…';
        try {
            const result = await window.pywebview.api.update_segmentation_mask(
                view.target_id, [{tool: completed.tool, radius: completed.radius, points: completed.points}], resultsPath, size.orientation
            );
            if (!result.success) throw new Error(result.error || 'Could not save mask');
            if (current()) {
                await setMask(result.segmentation);
                if (!current()) return;
                status.textContent = 'Saved';
                onSaved?.(result.segmentation);
            }
        } catch (error) {
            if (current()) { await setMask(saved); status.textContent = error.message; }
            else console.error('Mask autosave failed:', error);
        } finally { saving = false; }
    }
    const up = event => {
        if (stroke && event.pointerId === stroke.pointerId) {suppress(event); finish();}
        if (!pointerInside) {shift = backtick = false; applyMode();}
    };
    const leave = () => {
        pointerInside = false; lastPointer = null; ring.style.display = 'none';
        if (!stroke) {shift = backtick = false; applyMode();}
    };
    const wheel = event => {
        if (!mode || !event.shiftKey || !event.deltaY || !current()) return;
        suppress(event);
        radius = resizeMaskBrush(radius, event.deltaY);
        input.value = String(Math.round(radius * 2)); updateRing();
    };
    const contextMenu = event => {if (mode && current()) suppress(event);};
    const registrations = [['pointerenter', enter], ['pointerdown', down], ['pointermove', move], ['pointerup', up],
        ['pointercancel', up], ['pointerleave', leave], ['wheel', wheel], ['contextmenu', contextMenu]];
    for (const [name, handler] of registrations) surface.addEventListener(name, handler, {capture: true, passive: false});
    document.addEventListener('keydown', keydown, true);
    document.addEventListener('keyup', keyup, true);
    window.addEventListener('blur', blur);
    viewer.addHandler('animation', updateRing);
    viewer.addHandler('resize', updateRing);
    controller.zoomToBoxGeometry(size.bounds);
    return {
        isActive: () => !!mode,
        isSaving: () => saving || maskUpdating || !!stroke,
        setMask,
        dispose() {
            finish(); // Capture the original page and orientation even if navigation is underway.
            disposed = true;
            for (const [name, handler] of registrations) surface.removeEventListener(name, handler, true);
            document.removeEventListener('keydown', keydown, true);
            document.removeEventListener('keyup', keyup, true);
            window.removeEventListener('blur', blur);
            viewer.removeHandler('animation', updateRing); viewer.removeHandler('resize', updateRing);
            viewer.removeOverlay(shade); viewer.removeControl(bar);
            for (const button of toolButtons) {
                if (viewer.buttonGroup) {
                    const index = viewer.buttonGroup.buttons.indexOf(button);
                    if (index !== -1) viewer.buttonGroup.buttons.splice(index, 1);
                } else viewer.removeControl(button.element);
                button.element.remove();
                button.destroy();
            }
            bar.remove(); ring.remove();
            surface.style.cursor = previousCursor;
        },
    };
}

if (typeof module !== 'undefined' && module.exports) module.exports = {resizeMaskBrush, mountSegmentationEditor};
