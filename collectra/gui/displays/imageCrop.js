// Crop drawing controls, coordinate transforms and box annotations.
        // Draw-box toggle as an OpenSeadragon.Button — viewer.addButton() puts
        // it in the native zoom/home/fit ButtonGroup, matching their size,
        // spacing, group-hover fade and keyboard focus.
        //
        // OSD.Button only cycles rest/group/hover/down with no sticky state, so
        // syncDrawBoxButton() swaps all four layers between the normal and armed
        // sets; the armed set fills the rest layer too, so drawing mode shows
        // without a hover.
        //
        // Geometry follows OSD 5's nav PNGs (35x34, ~27px disc at 18,16.5).
        // Accent and armed fills read --app-primary/--app-primary-dark from the
        // live theme, so a project's theme.css carries through.
        let imageRotationBusy = false;
        let _drawBoxImgs = null;
        function drawBoxImages() {
            if (_drawBoxImgs) return _drawBoxImgs;
            const cs = getComputedStyle(document.documentElement);
            const primary = (cs.getPropertyValue('--app-primary') || '').trim() || '#f09226';
            const primaryDark = (cs.getPropertyValue('--app-primary-dark') || '').trim() || '#d97f18';
            const svg = (disc, ring, glyph) => 'data:image/svg+xml,' + encodeURIComponent(
                '<svg xmlns="http://www.w3.org/2000/svg" width="35" height="34" viewBox="0 0 35 34">' +
                '<circle cx="18" cy="16.5" r="13.5" fill="' + disc + '" stroke="' + ring + '" stroke-width="1"/>' +
                '<rect x="11.5" y="10" width="13" height="13" rx="2" fill="none" stroke="' + glyph + '" stroke-width="2"/>' +
                '</svg>');
            _drawBoxImgs = {
                normal: {
                    rest:  svg('rgba(244,244,244,0.62)', 'rgba(90,90,90,0.35)', 'rgba(40,40,40,0.62)'),
                    group: svg('#ececec', '#b6b6b6', '#333333'),
                    hover: svg('#ffffff', primary, primary),
                    down:  svg('#e4e4e4', '#8f8f8f', '#222222'),
                },
                armed: {
                    rest:  svg(primary, primaryDark, '#ffffff'),
                    group: svg(primary, primaryDark, '#ffffff'),
                    hover: svg(primaryDark, primaryDark, '#ffffff'),
                    down:  svg(primaryDark, primaryDark, '#ffffff'),
                },
            };
            return _drawBoxImgs;
        }

        function createDrawBoxButton(onClick) {
            const imgs = drawBoxImages().normal;
            return new OpenSeadragon.Button({
                tooltip: 'Draw box',
                srcRest: imgs.rest, srcGroup: imgs.group, srcHover: imgs.hover, srcDown: imgs.down,
                onClick: onClick,
            });
        }

        // Reverse sync: select the page box matching an artefact (if it's on this page).
        // Fraction of the box's own size added on each side when zooming to it,
        // so it doesn't sit flush against the viewer edges.
        const CROP_ZOOM_PADDING = 0.15;

        // Zooms to `geometry` — a starting point, not a locked frame; the
        // shared page viewer still allows free pan/zoom past it (e.g. to
        // check a crop's position on the page), same as any normal viewer.
        //
        // Uses fitBoundsWithConstraints, not plain fitBounds: fitBounds
        // explicitly ignores min/max zoom, so on a crop small enough that
        // fitting it requires more zoom than getMaxZoom() allows, it left the
        // viewport sitting past that ceiling — every subsequent zoom-in
        // request got silently clamped to the same (already-exceeded) ceiling,
        // i.e. zoom-in appeared dead.

        // Opening an ImageCrop node zooms the shared page viewer to just that
        // box; opening anything else that isn't a crop (e.g. the page's own
        // Image node) resets to the full page, so the previous crop's zoom
        // doesn't linger.

        // Called on pointerup and Enter/Escape (see global keydown listener below).




        // Redraws child crop boxes and segmentation masks on this panel's own
        // image — shared by the initial load and every create/update/
        // delete refresh, so they can't drift out of sync with each other.

        function calculateGeometryFromCropRegion(cropRegion, imgWidth, imgHeight) {
            if (!cropRegion) return null;
            if (imgWidth === undefined) [imgWidth, imgHeight] = getImageSize();
            let x = (cropRegion["x_center"] - cropRegion["width_relative"]/2)*imgWidth;
            let y = (cropRegion["y_center"] - cropRegion["height_relative"]/2)*imgHeight;
            let width = cropRegion["width_relative"]*imgWidth;
            let height = cropRegion["height_relative"]*imgHeight;
            return {x, y, width, height};
        }

        function calculateCropRegionFromGeometry(geometry, imgWidth, imgHeight) {
            if (imgWidth === undefined) [imgWidth, imgHeight] = getImageSize();
            const width_relative = geometry.w / imgWidth;
            const height_relative = geometry.h / imgHeight;
            const x_center = (geometry.x / imgWidth) + (width_relative / 2);
            const y_center = (geometry.y / imgHeight) + (height_relative / 2);
            return { x_center, y_center, width_relative, height_relative };
        }

        // Keeps a box's geometry inside [0, imgWidth] x [0, imgHeight] —
        // shrinks it first if it's larger than the image itself, then
        // slides it back in bounds.
        function clampGeometryToBounds(geometry, bounds) {
            const w = Math.min(geometry.w, bounds.w);
            const h = Math.min(geometry.h, bounds.h);
            const x = Math.min(Math.max(geometry.x, bounds.x), bounds.x + bounds.w - w);
            const y = Math.min(Math.max(geometry.y, bounds.y), bounds.y + bounds.h - h);
            return {
                ...geometry,
                x, y, w, h,
                bounds: { minX: x, minY: y, maxX: x + w, maxY: y + h },
            };
        }

        function clampGeometryToImage(geometry, imgWidth, imgHeight) {
            return clampGeometryToBounds(geometry, { x: 0, y: 0, w: imgWidth, h: imgHeight });
        }

        // Shared by the draw-time clamp and confirm-time parent detection so they can't disagree.
        function findContainingCrop(geometry, allAnnotations, excludeId) {
            const containing = allAnnotations.filter(ann => {
                if (ann.id === excludeId) return false;
                const g = ann.target.selector.geometry;
                return (geometry.x > g.x && geometry.y > g.y &&
                        (geometry.x + geometry.w) < (g.x + g.w) &&
                        (geometry.y + geometry.h) < (g.y + g.h));
            });
            return containing.length > 0 ? containing[containing.length - 1] : null;
        }

        function rectsOverlap(a, b) {
            return a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;
        }

        // The new box's parent. Prefers whichever crop's own tab is open,
        // but only when the drawn box actually touches it — the view is
        // allowed to show a little past the open crop's own bounds (see
        // zoomToBoxGeometry's padding), so a box drawn mostly inside it but
        // poking slightly past its exact edge should still parent to it
        // rather than falling through to findContainingCrop's strict "fully
        // inside" test. But the crop's tab being open doesn't mean every box
        // you draw anywhere belongs to it — you can zoom back out within that
        // same tab and draw something unrelated elsewhere on the page, and
        // that must NOT get clamped into the open crop's small bounds just
        // because its tab happens to still be selected.
        function findParentCrop(controller, geometry, allAnnotations, excludeId) {
            if (controller && controller.selectedNodeId) {
                const openCrop = allAnnotations.find((a) =>
                    a.id !== excludeId && artefactMatchesBox(controller.selectedNodeId, (a.bodies[0] && a.bodies[0].id) || a.id));
                if (openCrop && rectsOverlap(geometry, openCrop.target.selector.geometry)) return openCrop;
            }
            return findContainingCrop(geometry, allAnnotations, excludeId);
        }

        async function updateAnnotationCoordinates(annotationId, geometry) {
            const cropRegion = calculateCropRegionFromGeometry(geometry);
            try {
                const result = await window.pywebview.api.update_node_coordinates(annotationId, cropRegion);
                if (!result.success) {
                    console.error('Failed to update annotation coordinates:', result.error);
                    return;
                }
                const display_data = result.rows.filter(row => row.type === "collectra.ImageCrop");
                // Reload annotations on the image
                anno.clearAnnotations();
                display_data.forEach(row => {
                    const geom = calculateGeometryFromCropRegion(row.crop_region);
                    if (!geom) return;
                    anno.addAnnotation(createAnnotation(geom, row));
                });
            } catch (e) {
                console.error('Error updating annotation coordinates:', e);
            }
        }

        function createAnnotation(geometry, data) {
            let annotation = {
                id: data.id,
                target:{
                    selector:{
                        type: 'RECTANGLE',
                        geometry: {
                            bounds: {
                                minX: geometry.x,
                                minY: geometry.y,
                                maxX: geometry.x + geometry.width,
                                maxY: geometry.y + geometry.height
                            },
                            x: geometry.x,
                            y: geometry.y,
                            w: geometry.width,
                            h: geometry.height
                        }
                    }
                },
                bodies:[
                    data
                ]
            };
            return annotation;
        }

        // ── PAGE ARTEFACT VIEWER (middle panel) ───────────────────────────────
        // Shows the highlighted page's source image with its ImageCrop bounding boxes,
        // read-only. Driven by the left-hand page selection (not the DAG).
        // pageViewer/pageAnno/easyMDE/selectedNodeId used to live here as
        // single globals; each is now a PanelController instance field (see
        // below) so two panels can hold independent state simultaneously.
        let lastMousePos = { x: 0, y: 0 };
        let pageMouseTracked = false;

        function ensureMouseTracking() {
            if (pageMouseTracked) return;
            document.addEventListener('mousemove', (e) => {
                lastMousePos = { x: e.clientX, y: e.clientY };
                const t = document.getElementById('pageBoxTooltip');
                if (t && t.style.display === 'block') {
                    t.style.left = (lastMousePos.x + 14) + 'px';
                    t.style.top  = (lastMousePos.y + 14) + 'px';
                }
            });
            pageMouseTracked = true;
        }

        function hideBoxTooltip() {
            const t = document.getElementById('pageBoxTooltip');
            if (t) t.style.display = 'none';
        }

        function getBoxTooltip() {
            let t = document.getElementById('pageBoxTooltip');
            if (!t) {
                t = document.createElement('div');
                t.id = 'pageBoxTooltip';
                document.body.appendChild(t);
            }
            return t;
        }

        // e.g. "main_text-ba7d0d26" → "Main Text", "sandglass1" → "Sandglass".
        function prettyCropName(id) {
            return cropBaseName(id)
                .replace(/_/g, ' ')
                .trim()
                .replace(/\b\w/g, c => c.toUpperCase());
        }

        // Strips a trailing uuid (old ids) or number (current scheme).
        function cropBaseName(id) {
            const str = String(id);
            const legacyStripped = str.replace(/-[0-9a-f]{6,}$/i, '');
            if (legacyStripped !== str) return legacyStripped;
            return str.replace(/\d+$/, '');
        }

        // A bare label (e.g. "main_text_image", no uuid of its own — a DAG
        // click) matches any box sharing that base name. A real per-instance
        // id (already has its own uuid, e.g. "primary_specimen_label-3931e04c")
        // must match exactly — otherwise every instance of a multi-instance
        // label prefix-matches the same first box, regardless of tab.
        function artefactMatchesBox(artefactId, boxId) {
            const a = String(artefactId);
            const b = String(boxId);
            if (a === b) return true;
            return cropBaseName(a) === a && a === cropBaseName(b);
        }
