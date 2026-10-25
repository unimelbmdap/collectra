// Image viewer, zoom, rotation, crop interactions and annotation refresh.
function installImageDisplay(PanelController) {
        PanelController.prototype.clearImageDisplay = function() {
            this._maskEditor?.dispose();
            this._maskEditor = null;
            this._segmentationHoverCleanup?.();
            this._segmentationHoverCleanup = null;
            this._shiftDrawCleanup?.();
            this._shiftDrawCleanup = null;
            if (this.pageAnno)   { try { this.pageAnno.destroy(); }   catch (e) {} this.pageAnno = null; }
            if (this.pageViewer) { try { this.pageViewer.destroy(); } catch (e) {} this.pageViewer = null; }
            this._segmentationOverlays = [];
            if (this.el.imageDisplay) { this.el.imageDisplay.style.display = 'none'; this.el.imageDisplay.innerHTML = ''; }
            this.drawBoxBtn = null; // owned by the viewer's ButtonGroup, destroyed by pageViewer.destroy() above
            const tooltip = document.getElementById('pageBoxTooltip'); // shared across panels
            if (tooltip) tooltip.style.display = 'none';
        };
        PanelController.prototype.zoomToBoxGeometry = function(geometry) {
            if (!this.pageViewer || !geometry) return;
            const viewport = this.pageViewer.viewport;
            const rect = viewport.imageToViewportRectangle(geometry.x, geometry.y, geometry.w, geometry.h);
            const padX = rect.width * CROP_ZOOM_PADDING;
            const padY = rect.height * CROP_ZOOM_PADDING;
            const padded = new OpenSeadragon.Rect(
                rect.x - padX, rect.y - padY, rect.width + 2 * padX, rect.height + 2 * padY
            );
            viewport.fitBoundsWithConstraints(padded);
        };
        PanelController.prototype.highlightPageBox = function(artefactId) {
            if (!this.pageAnno) return;
            const anns = this.pageAnno.getAnnotations ? this.pageAnno.getAnnotations() : [];
            const match = anns.find(a => artefactMatchesBox(artefactId, (a.bodies && a.bodies[0] && a.bodies[0].id) || a.id));
            if (match) {
                this.pageAnno.setSelected?.(match.id);
                this.zoomToBoxGeometry(match.target.selector.geometry);
            } else {
                this.pageAnno.cancelSelected?.();
                if (this.displayView?.segmentation) this.zoomToBoxGeometry(segmentationBoundsInPreview(this, this.displayView.segmentation));
                else if (this.pageViewer) this.pageViewer.viewport.goHome();
            }
        };
        PanelController.prototype.flushAnnotationUpdate = function() {
            if (!this._updateAnnotationFlush) return;
            clearTimeout(this._updateAnnotationTimer);
            return this._updateAnnotationFlush();
        };
        PanelController.prototype.syncDrawBoxButton = function() {
            const inFullPage = !!(this.pageViewer && this.pageViewer.isFullPage());
            // A leaf label (no ImageCrop children downstream, e.g. Hespi's
            // "scale"/"swatch") has nothing a drawn box could ever be
            // confirmed as — block drawing itself, not just the popup's
            // Confirm button. Skipped when the open node can't be resolved to
            // a DAG label at all (nothing to judge against yet, e.g. during
            // initial load), so drawing isn't blocked speculatively.
            const focused = this.selectedNodeId ? resolveDagLabel(this.selectedNodeId) : null;
            const noValidChildType = !inFullPage && !!focused && cropTypeOptionsForController(this).labels.length === 0;
            const maskEditing = !!this._maskEditor?.isActive();
            const disabled = inFullPage || noValidChildType || !this.displayView?.can_create_crop || maskEditing;
            const armed = !disabled && !!(this.pageAnno && this.pageAnno.isDrawingEnabled());
            if (noValidChildType && this.pageAnno && this.pageAnno.isDrawingEnabled()) {
                this.pageAnno.setDrawingEnabled(false);
            }
            // OSD's own drag-to-pan is unconditional and reads this live on every
            // drag (see its gestureSettingsByDeviceType) — nothing about
            // drawingEnabled suppresses it on its own, so a drag while armed was
            // panning the view under the crosshair instead of drawing a box, unlike
            // a dedicated draw tool in an image editor. Suppressed only while armed.
            if (this.pageViewer) this.pageViewer.gestureSettingsMouse.dragToPan = !armed && !maskEditing;
            const btn = this.drawBoxBtn;
            if (btn) {
                const set = armed ? drawBoxImages().armed : drawBoxImages().normal;
                if (btn.imgRest)  btn.imgRest.src  = set.rest;
                if (btn.imgGroup) btn.imgGroup.src = set.group;
                if (btn.imgHover) btn.imgHover.src = set.hover;
                if (btn.imgDown)  btn.imgDown.src  = set.down;
                // OSD.Button sets element.title = 'Draw box' at construction and
                // never touches it again — override it only for the reason it's
                // disabled; leave OSD's own default alone otherwise.
                btn.element.title = noValidChildType ? 'No children for this label' : 'Draw box';
                // disable() also stops tracking and fades to 0.2; enable() forces
                // the button back to GROUP state — so only flip on a real change.
                if (disabled && !btn.element.disabled) btn.disable();
                else if (!disabled && btn.element.disabled) btn.enable();
            }
            if (this.el.imageDisplay) this.el.imageDisplay.classList.toggle('draw-box-cursor', armed);
        };
        PanelController.prototype.showPageImage = function() {
            this.stopTextAutosave();
            if (this.el.imageDisplay) this.el.imageDisplay.style.display = 'block';
            if (this.el.textDisplay)  { this.el.textDisplay.style.display = 'none'; this.el.textDisplay.innerHTML = ''; }
            // Clear any leftover content from a previously-selected Text node —
            // otherwise its stale badge/note sits underneath the photo.
            if (this.el.contentDiv)   { this.el.contentDiv.style.display = 'none'; this.el.contentDiv.innerHTML = ''; }
            this.clearHeader();
            if (this.easyMDE) { try { this.easyMDE.toTextArea(); this.easyMDE = null; } catch (e) {} }
        };
        PanelController.prototype.renderPageArtefact = async function(imagePath, view = null, identity = null) {
            const display     = this.el.imageDisplay;
            const placeholder = this.el.placeholder;
            const contentDiv  = this.el.contentDiv;
            if (!display || !apiReady || (!imagePath && !view)) return;

            this.currentImagePath = imagePath;
            this.stopTextAutosave();
            if (this.el.textDisplay) this.el.textDisplay.style.display = 'none';
            if (this.easyMDE) { this.easyMDE.toTextArea(); this.easyMDE = null; }
            this.clearPageArtefact();
            // Supersede any earlier render still mid-await: each await below bails if
            // a newer call (or a clearPageArtefact) has replaced this token. Without
            // this, two overlapping renders build two viewers → a doubled draw-box button.
            const token = (this._artefactToken = {});
            if (placeholder) placeholder.style.display = 'none';
            if (contentDiv)  contentDiv.style.display  = 'none';
            this.clearHeader();
            if (identity) {
                this.el.headerId.textContent = identity.id;
            }
            display.style.display = 'block';

            try {
                if (!view) {
                    const result = await window.pywebview.api.get_artefact_display();
                    if (!result.success) throw new Error(result.error);
                    view = result.view;
                }
                if (this._artefactToken !== token) return;
                this.displayView = view;
                this.currentImagePath = view.image_path;
                const imgResult = { success: true, data: view.source };
                if (this._artefactToken !== token) return;
                if (!imgResult.success) {
                    display.innerHTML = '<div class="image-placeholder">Could not load page image.</div>';
                    return;
                }
                display.innerHTML = '';
                await new Promise((resolve, reject) => {
                    this.pageViewer = OpenSeadragon({
                        element: display,
                        prefixUrl: "openseadragon/images/",
                        drawer: 'canvas', // avoids exhausting WebKit's WebGL context limit across page switches
                        tileSources: { type: 'image', url: imgResult.data },
                        gestureSettingsMouse: { clickToZoom: false, dblClickToZoom: false },
                    });
                    this.pageViewer.addHandler('open', resolve);
                    this.pageViewer.addHandler('open-failed', event => reject(new Error(event.message || 'Image preview failed')));
                });
                // A newer render's clearPageArtefact has already torn down the viewer
                // we just built (it runs before we can resume here) — just stop.
                if (this._artefactToken !== token) return;

                // Just disarms on entry — the real guards are the isFullPage() checks below.
                this.pageViewer.addHandler('full-page', (event) => {
                    if (event.fullPage) this.pageAnno.setDrawingEnabled(false);
                    this.syncDrawBoxButton();
                });

                this.pageAnno = AnnotoriousOSD.createOSDAnnotator(this.pageViewer, { drawingEnabled: false, drawingMode: 'drag' });

                const drawBoxBtn = createDrawBoxButton(() => {
                    if (this.pageViewer.isFullPage()) return; // live check
                    this.pageAnno.setDrawingEnabled(!this.pageAnno.isDrawingEnabled());
                    this.syncDrawBoxButton();
                });
                // addButton() appends it to the native zoom/home/fit ButtonGroup;
                // fall back to a standalone control if nav buttons are disabled.
                if (this.pageViewer.buttonGroup) {
                    this.pageViewer.addButton(drawBoxBtn);
                } else {
                    this.pageViewer.addControl(drawBoxBtn.element, { anchor: OpenSeadragon.ControlAnchor.TOP_LEFT });
                }
                this.drawBoxBtn = drawBoxBtn;
                this.syncDrawBoxButton();
                this._shiftDrawCleanup = mountShiftDraw({
                    element: display, document, window,
                    canDraw: () => {
                        this.syncDrawBoxButton();
                        return !view.segmentation && this._artefactToken === token && !drawBoxBtn.element.disabled
                            && !imageRotationBusy && !document.querySelector('.modal-overlay.visible');
                    },
                    isDrawing: () => this.pageAnno.isDrawingEnabled(),
                    setDrawing: enabled => this.pageAnno.setDrawingEnabled(enabled),
                    sync: () => this.syncDrawBoxButton(),
                });
                if (view.can_rotate) {
                    const rotateIcon = 'data:image/svg+xml,' + encodeURIComponent(
                        '<svg xmlns="http://www.w3.org/2000/svg" width="35" height="34" viewBox="0 0 35 34">' +
                        '<circle cx="18" cy="16.5" r="13.5" fill="#ececec" stroke="#b6b6b6"/>' +
                        '<path d="M24 14a7 7 0 1 0 0 6M24 9v6h-6" fill="none" stroke="#333" stroke-width="2"/>' +
                        '</svg>');
                    const rotateButton = new OpenSeadragon.Button({
                        tooltip: 'Rotate 90° clockwise',
                        srcRest: rotateIcon, srcGroup: rotateIcon, srcHover: rotateIcon, srcDown: rotateIcon,
                        onClick: async () => {
                            if (this._artefactToken !== token || imageRotationBusy) return;
                            imageRotationBusy = true;
                            rotateButton.disable();
                            try {
                                await Promise.all(Object.values(panelControllers).map(controller => controller.flushAnnotationUpdate()));
                                if (this._artefactToken !== token) return;
                                const result = await window.pywebview.api.rotate_image_clockwise(view.target_id);
                                if (!result.success) throw new Error(result.error);
                                await Promise.all(Object.values(panelControllers).map(async controller => {
                                    const current = controller.displayView;
                                    if (!current || !result.updated_ids.includes(current.target_id)) return;
                                    const refreshed = await window.pywebview.api.get_artefact_display(current.target_id, current.rgb_view_index || 0);
                                    if (!refreshed.success) throw new Error(refreshed.error);
                                    if (controller.displayView === current) await controller.renderPageArtefact(current.image_path, refreshed.view, refreshed);
                                }));
                            } catch (error) {
                                alert(error.message || String(error));
                            } finally {
                                imageRotationBusy = false;
                                if (this._artefactToken === token) rotateButton.enable();
                            }
                        },
                    });
                    if (this.pageViewer.buttonGroup) this.pageViewer.addButton(rotateButton);
                    else this.pageViewer.addControl(rotateButton.element, {anchor: OpenSeadragon.ControlAnchor.TOP_LEFT});
                }

                this.pageAnno.setStyle(cropAnnotationStyle);

                // Hover tooltip; also feeds the right-click delete menu
                // below — Annotorious's contextmenu event carries no
                // annotation reference.
                ensureMouseTracking();
                const tooltip = getBoxTooltip();
                let hoveredAnnotation = null;
                this.pageAnno.on('mouseEnterAnnotation', (annotation) => {
                    hoveredAnnotation = annotation;
                    if (document.querySelector('.modal-overlay.visible')) return; // no hover over a dialog
                    const body = (annotation.bodies && annotation.bodies[0]) || {};
                    const rawId = body.id || annotation.id || '';
                    const typeName = (body.type || 'collectra.ImageCrop');
                    // A name equal to the id is treated as unset.
                    const displayName = (body.name && body.name !== rawId) ? escapeHtml(body.name) : prettyCropName(rawId);
                    const confidence = typeof body.confidence === 'number' && Number.isFinite(body.confidence)
                        ? `<div>Confidence: ${(body.confidence * 100).toFixed(1)}%</div>`
                        : '';
                    tooltip.innerHTML =
                        `<div class="box-name">${displayName}</div>` +
                        `<div>${typeName}</div>` +
                        confidence +
                        `<div class="box-id">${rawId}</div>`;
                    tooltip.style.display = 'block';
                    tooltip.style.left = (lastMousePos.x + 14) + 'px';
                    tooltip.style.top  = (lastMousePos.y + 14) + 'px';
                    scheduleHoverHide(tooltip);
                });
                this.pageAnno.on('mouseLeaveAnnotation', () => {
                    hoveredAnnotation = null;
                    cancelHoverHide(tooltip);
                    tooltip.style.display = 'none';
                });
                display.addEventListener('contextmenu', (event) => {
                    if (!hoveredAnnotation) return;
                    event.preventDefault();
                    showAnnotationContextMenu(event, hoveredAnnotation);
                });

                // New box drawn: clamp, then ask for a label. Mutating the
                // geometry alone doesn't reach the live view — needs updateAnnotation() too.
                this.pageAnno.on('createAnnotation', (annotation) => {
                    if (this.pageViewer.isFullPage()) { this.pageAnno.removeAnnotation(annotation.id); return; }
                    const [imgW, imgH] = [this.pageViewer.world._contentSize.x, this.pageViewer.world._contentSize.y];
                    // Stay inside a containing box, not just the image — it's about to become its parent.
                    const parentCrop = findParentCrop(this, annotation.target.selector.geometry, this.pageAnno.getAnnotations(), annotation.id);
                    const bounds = parentCrop && !this.displayView
                        ? parentCrop.target.selector.geometry
                        : { x: 0, y: 0, w: imgW, h: imgH };
                    annotation.target.selector.geometry = clampGeometryToBounds(annotation.target.selector.geometry, bounds);
                    showLabelPopup(annotation, this); // sets pendingAnnotation before updateAnnotation()'s guard needs it
                    this.pageAnno.updateAnnotation(annotation);
                });

                // Existing box dragged/resized: clamp, persist, redraw. Fires on every
                // pointermove, not just release — debounced so it doesn't tear down
                // the box mid-gesture. Instance properties, not closure locals, so
                // flushAnnotationUpdate() (Enter/Escape below) can reach this panel.
                this._updateAnnotationTimer = null;
                this._updateAnnotationFlush = null;
                const queueUpdate = (updated, { debounce }) => {
                    if (pendingAnnotation && pendingAnnotation.id === updated.id) return; // still mid-create, nothing to persist
                    if (this._pushingClampedUpdate) return; // our own re-push below, not a new edit
                    this._updateAnnotationFlush = async () => {
                        this._updateAnnotationFlush = null;
                        const [imgW, imgH] = [this.pageViewer.world._contentSize.x, this.pageViewer.world._contentSize.y];
                        const clamped = clampGeometryToImage(updated.target.selector.geometry, imgW, imgH);
                        // Pushed immediately, not after the round-trip below, or the box
                        // visibly sits unclamped until that resolves.
                        updated.target.selector.geometry = clamped;
                        this._pushingClampedUpdate = true;
                        this.pageAnno.updateAnnotation(updated);
                        this._pushingClampedUpdate = false;
                        const cropRegion = calculateCropRegionFromGeometry(clamped, imgW, imgH);
                        try {
                            const result = await window.pywebview.api.update_node_coordinates(updated.id, cropRegion, this.displayView?.target_id || '');
                            if (result.success) Object.values(panelControllers).forEach((c) => c.refreshPageAnnotations(result.rows));
                            else console.error('Failed to update annotation coordinates:', result.error);
                        } catch (e) {
                            console.error('Error updating annotation coordinates:', e);
                        }
                    };
                    clearTimeout(this._updateAnnotationTimer);
                    if (debounce) this._updateAnnotationTimer = setTimeout(this._updateAnnotationFlush, 250);
                };
                // Annotorious only emits 'updateAnnotation' when the box is deselected, so
                // watch the store for live edits and commit them on pointerup (no timer —
                // re-pushing mid-gesture would tear down the box being dragged).
                this.pageAnno.on('updateAnnotation', (updated) => queueUpdate(updated, { debounce: true }));
                this.pageAnno.state.store.observe((event) => {
                    (event.changes.updated || []).forEach(({ oldValue, newValue }) => {
                        const before = oldValue.target.selector.geometry;
                        const after = newValue.target.selector.geometry;
                        if (before && after && JSON.stringify(before) !== JSON.stringify(after)) queueUpdate(newValue, { debounce: false });
                    });
                }, { origin: 'LOCAL' });
                display.addEventListener('pointerup', () => this.flushAnnotationUpdate());

                // The display response already contains these annotations.
                await this.refreshPageAnnotations(null, view.annotations || []);
                if (this._artefactToken !== token) return;

                if (view.segmentation) {
                    const editor = await mountSegmentationEditor(this, view, {
                        token, resultsPath: selectedYamlPath,
                        onSaved: segmentation => {
                            this.displayView.segmentation = segmentation;
                            for (const controller of Object.values(panelControllers)) {
                                if (controller.displayView?.image_path !== view.image_path) continue;
                                controller.refreshPageAnnotations(null).catch(console.error);
                                if (controller !== this && controller.displayView.target_id === view.target_id
                                    && controller._maskEditor && !controller._maskEditor.isSaving()) {
                                    controller.displayView.segmentation = segmentation;
                                    controller._maskEditor.setMask(segmentation).catch(console.error);
                                }
                            }
                        },
                    });
                    if (this._artefactToken !== token) { editor?.dispose(); return; }
                    this._maskEditor = editor;
                }

                const viewer = this.pageViewer;
                mountRgbViewControls({
                    viewer, view,
                    isCurrent: () => this._artefactToken === token,
                    loadView: index => window.pywebview.api.get_artefact_display(view.target_id, index),
                    applyView: nextView => new Promise((resolve, reject) => {
                        const bounds = viewer.viewport.getBounds(true);
                        // Replace the pixels in the existing viewer: keep zoom,
                        // pan, the annotator, and all child bounding boxes.
                        viewer.addTiledImage({
                            tileSource: {type: 'image', url: nextView.source},
                            index: 0,
                            replace: true,
                            success: () => {
                                if (this._artefactToken === token) {
                                    viewer.viewport.fitBounds(bounds, true);
                                    this.displayView = nextView;
                                }
                                resolve();
                            },
                            error: event => reject(new Error(event.message || 'Could not display RGB view')),
                        });
                    }),
                });
            } catch (e) {
                if (this._artefactToken === token) display.textContent = e.message;
                console.error('Error rendering page artefact:', e);
            }
        };
        PanelController.prototype.refreshPageAnnotations = async function(rows, initialAnnotations = null) {
            if (!this.pageAnno || !this.pageViewer || !this.displayView) return;
            const token = this._artefactToken;
            const refreshToken = (this._maskRefreshToken = {});
            const result = initialAnnotations !== null
                ? {success: true, rows: initialAnnotations}
                : this.displayView.can_create_crop
                    ? await window.pywebview.api.get_display_annotations(this.displayView.target_id)
                    : {success: true, rows: this.displayView.annotations || []};
            if (token !== this._artefactToken || refreshToken !== this._maskRefreshToken || !result.success || !this.pageAnno) return;
            const imgW = this.pageViewer.world._contentSize.x;
            const imgH = this.pageViewer.world._contentSize.y;
            this.pageAnno.clearAnnotations();
            for (const element of this._segmentationOverlays || []) this.pageViewer.removeOverlay(element);
            this._segmentationOverlays = [];
            this._segmentationHoverCleanup?.();
            this._segmentationHoverCleanup = mountSegmentationHover(this, result.rows, {token, imgW, imgH});
            result.rows.forEach(r => {
                const geom = calculateGeometryFromCropRegion(r.crop_region, imgW, imgH);
                if (geom) this.pageAnno.addAnnotation(createAnnotation(geom, r));
            });
            await renderSegmentationOverlays(this, result.rows, {token, refreshToken, imgW, imgH});
        };
}

async function renderImageArtefact(controller, view, result) {
    return controller.renderPageArtefact(view.image_path, view, result);
}

if (typeof module !== "undefined" && module.exports) {
    module.exports = {installImageDisplay, renderImageArtefact};
} else {
    registerArtefactRenderer("image", renderImageArtefact);
}

// Compatibility helpers for the original single-image view.
        async function displayImageFromPath(imagePath) {
            try {
                clearImage();
                const result = await window.pywebview.api.get_image_base64(imagePath);
                if (result.success) {
                    // #imageContainer lives inside bottom panel's markup — gone if bottom's been removed
                    const container = document.getElementById('imageContainer');
                    if (!container) return;
                    container.innerHTML = '';

                    // Wait for OpenSeadragon to fully load the image
                    await new Promise((resolve) => {
                        viewer = OpenSeadragon({
                            element: container,
                            prefixUrl: "openseadragon/images/",
                            // Default drawer is WebGL; every new instance opens a fresh
                            // WebGL context, and WebKit (pywebview's engine on macOS)
                            // doesn't reliably reclaim one the instant destroy() runs —
                            // switching pages/projects repeatedly exhausts its context
                            // limit ("too many active WebGL contexts"). Plain 2D canvas
                            // is all this simple pan/zoom viewer needs anyway.
                            drawer: 'canvas',
                            tileSources: {
                                type: 'image',
                                url: result.data
                            },
                            gestureSettingsMouse: {
                                clickToZoom: false,
                                dblClickToZoom: false,
                                dblClickDragToZoom: false,
                            }
                        });
                        viewer.addHandler('open', function() {
                            console.log('Image loaded into OpenSeadragon');
                            resolve();
                        });
                    });

                    anno = AnnotoriousOSD.createOSDAnnotator(viewer, {
                        drawingEnabled: true,
                        drawingMode: 'drag',
                    });

                    anno.setStyle(cropAnnotationStyle);

                    anno.on('selectionChanged', function(annotations, originalEvent){
                        // Grid removed — selection sync dormant until editing is rebuilt
                    });
                    anno.on('updateAnnotation', function(updated, previous){
                        const geometry = updated.target.selector.geometry;
                        updateAnnotationCoordinates(updated.id, geometry);
                    });
                    anno.on('createAnnotation', function(annotation) {
                        // Show label popup and store the annotation temporarily
                        showLabelPopup(annotation);
                    });
                    currentImagePath = imagePath;
                } else {
                    console.error('Failed to load image:', result.error);
                    clearImage();
                }
            } catch (e) {
                console.error('Error loading image:', e);
                clearImage();
            }
        }

        function clearImage() {
            // #imageContainer lives inside bottom panel's markup — gone if bottom's been removed
            const container = document.getElementById('imageContainer');
            if (container) container.innerHTML = '<div class="image-placeholder">No image loaded. Select a folder containing an image file.</div>';
            if (anno) {
                anno.clearAnnotations();
                anno.destroy();
                anno = null;
            }
            viewer?.destroy();
            viewer = null;
            currentImagePath = "";
        }

        function getImageSize(){
            return [viewer.world._contentSize.x, viewer.world._contentSize.y];
        }
