// Shared pointer hit testing and tab navigation for child crops and masks.
function childImageAtPointer(controller, rows, event, imgW, imgH) {
    const viewer = controller.pageViewer;
    const item = viewer?.world.getItemAt(0);
    if (!item) return null;
    const rect = viewer.container.getBoundingClientRect();
    const pixel = new OpenSeadragon.Point(event.clientX - rect.left, event.clientY - rect.top);
    const point = item.viewportToImageCoordinates(viewer.viewport.pointFromPixel(pixel, true));
    const x = point.x / imgW, y = point.y / imgH;
    const bounds = row => {
        if (row.mask_region) return row.mask_region;
        const crop = row.crop_region;
        if (!crop) return null;
        return {x: crop.x_center - crop.width_relative / 2, y: crop.y_center - crop.height_relative / 2,
            width: crop.width_relative, height: crop.height_relative};
    };
    return rows.map(row => ({row, box: bounds(row)})).filter(({box: r}) =>
        r && x >= r.x && x <= r.x + r.width && y >= r.y && y <= r.y + r.height
    ).sort((a, b) => a.box.width * a.box.height - b.box.width * b.box.height)[0]?.row || null;
}

function mountChildImageNavigation(controller, rows, {token, imgW, imgH}) {
    const element = controller.el.imageDisplay;
    const open = event => {
        if (controller._artefactToken !== token || document.querySelector('.modal-overlay.visible')
            || controller.pageAnno?.isDrawingEnabled?.() || event.button !== 0) return;
        const match = childImageAtPointer(controller, rows, event, imgW, imgH);
        if (!match) return;
        event.preventDefault();
        event.stopPropagation();
        controller.openInNewTab(match.id);
    };
    element.addEventListener('dblclick', open, true);
    return () => element.removeEventListener('dblclick', open, true);
}
