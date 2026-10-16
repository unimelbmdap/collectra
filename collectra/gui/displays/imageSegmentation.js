// Source-aligned instance masks, coloured with their pipeline DAG label.
function mountSegmentationHover(controller, rows, {token, imgW, imgH}) {
    const element = controller.el?.imageDisplay;
    if (!element) return () => {};
    const candidates = rows.filter(row => row.mask_region);
    let hovered = null;
    const hide = () => {
        if (!hovered) return;
        const tooltip = getBoxTooltip();
        cancelHoverHide(tooltip);
        tooltip.style.display = 'none';
        hovered = null;
    };
    const move = event => {
        if (controller._artefactToken !== token || document.querySelector('.modal-overlay.visible')
            || controller.pageAnno?.isDrawingEnabled?.()) {
            hide();
            return;
        }
        const viewer = controller.pageViewer;
        const item = viewer?.world.getItemAt(0);
        if (!item) return;
        const rect = viewer.container.getBoundingClientRect();
        const pixel = new OpenSeadragon.Point(event.clientX - rect.left, event.clientY - rect.top);
        const point = item.viewportToImageCoordinates(viewer.viewport.pointFromPixel(pixel, true));
        const x = point.x / imgW, y = point.y / imgH;
        // Prefer the smallest box when neighbouring or nested instances overlap.
        const match = candidates.filter(({mask_region: r}) =>
            x >= r.x && x <= r.x + r.width && y >= r.y && y <= r.y + r.height
        ).sort((a, b) => a.mask_region.width * a.mask_region.height - b.mask_region.width * b.mask_region.height)[0];
        if (!match) { hide(); return; }
        const tooltip = getBoxTooltip();
        if (hovered !== match) {
            hovered = match;
            tooltip.textContent = match.name || match.id;
            tooltip.style.display = 'block';
            scheduleHoverHide(tooltip);
        }
        tooltip.style.left = (event.clientX + 14) + 'px';
        tooltip.style.top = (event.clientY + 14) + 'px';
    };
    element.addEventListener('pointermove', move);
    element.addEventListener('pointerleave', hide);
    return () => {
        element.removeEventListener('pointermove', move);
        element.removeEventListener('pointerleave', hide);
        hide();
    };
}

async function renderSegmentationOverlays(controller, rows, {token, refreshToken, imgW, imgH}) {
            await Promise.all(rows.filter(r => r.mask_source).map(async r => {
                const mask = new Image();
                mask.src = r.mask_source;
                await mask.decode();
                if (token !== controller._artefactToken || refreshToken !== controller._maskRefreshToken) return;
                const canvas = document.createElement('canvas');
                canvas.width = mask.naturalWidth;
                canvas.height = mask.naturalHeight;
                const context = canvas.getContext('2d');
                context.drawImage(mask, 0, 0);
                context.globalCompositeOperation = 'source-in';
                context.fillStyle = nodeColor(r.label || r.id);
                context.fillRect(0, 0, canvas.width, canvas.height);
                canvas.style.opacity = '0.35';
                canvas.style.pointerEvents = 'none';
                const region = r.mask_region;
                const item = controller.pageViewer.world.getItemAt(0);
                controller.pageViewer.addOverlay({element: canvas, location: item.imageToViewportRectangle(
                    region.x * imgW, region.y * imgH, region.width * imgW, region.height * imgH
                ), checkResize: false});
                controller._segmentationOverlays.push(canvas);
            }));
}
