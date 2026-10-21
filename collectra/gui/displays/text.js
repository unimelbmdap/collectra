// Text, Markdown and XML editors, autosave and editor toolbar lifecycle.

// Text size is one app-wide setting: a CSS variable on the document scales
// the source and preview in every panel, and is remembered between sessions.
const TEXT_ZOOM = {min: 0.6, max: 2.5, step: 0.1, key: 'collectra.textZoom'};

function stepTextZoom(current, direction) {
    const next = Math.round((current + direction * TEXT_ZOOM.step) * 10) / 10;
    return Math.min(TEXT_ZOOM.max, Math.max(TEXT_ZOOM.min, next));
}

function storedTextZoom() {
    try {
        const value = parseFloat(localStorage.getItem(TEXT_ZOOM.key));
        return Number.isFinite(value) ? Math.min(TEXT_ZOOM.max, Math.max(TEXT_ZOOM.min, value)) : 1;
    } catch (e) {
        return 1;
    }
}

function applyTextZoom(zoom) {
    document.documentElement.style.setProperty('--text-zoom', zoom);
    try { localStorage.setItem(TEXT_ZOOM.key, String(zoom)); } catch (e) {}
    for (const label of document.querySelectorAll('.text-zoom-level')) {
        label.textContent = `${Math.round(zoom * 100)}%`;
    }
    // CodeMirror measures line heights, so it must re-measure after a font change.
    for (const element of document.querySelectorAll('.CodeMirror')) element.CodeMirror?.refresh();
}

function installTextDisplay(PanelController) {
        PanelController.prototype.renderTextNodeAsPage = async function(nodeId, detail, view) {
            const { placeholder, imageDisplay, contentDiv, headerId, textDisplay } = this.el;
            if (!textDisplay) return;

            if (placeholder) placeholder.style.display = 'none';
            if (imageDisplay) imageDisplay.style.display = 'none';
            if (contentDiv)  { contentDiv.style.display = 'none'; contentDiv.innerHTML = ''; }
            textDisplay.style.display = 'flex';
            textDisplay.innerHTML = '<div class="book-page"><div class="book-page-label">Loading…</div></div>';

            const isTei = view.format === 'xml';
            headerId.textContent   = nodeId;

            const mountEditor = (text, tei) => {
                this.stopTextAutosave();
                if (this.easyMDE) { try { this.easyMDE.toTextArea(); this.easyMDE = null; } catch (e) {} }
                // Classes, not ids — this markup can exist in two panels at
                // once, so a fixed id here would duplicate across the document.
                this.clearNodeTools();
                textDisplay.innerHTML = `<textarea class="mde-textarea"></textarea>`;
                this.easyMDE = new EasyMDE({
                    autoDownloadFontAwesome: false,
                    element: textDisplay.querySelector('.mde-textarea'),
                    initialValue: text,
                    spellChecker: false,
                    autofocus: false,
                    status: false,
                    sideBySideFullscreen: false,
                    // EasyMDE defaults minHeight to "300px" and applies it as an
                    // inline min-height directly on .CodeMirror-scroll, which
                    // fights our flex-based sizing (forces the scroller taller
                    // than its actual available space, clipped by .CodeMirror's
                    // overflow:hidden, and throws off CodeMirror's own scroll
                    // math so wheel-scrolling silently does nothing). "0" lets
                    // the flex layout be the only thing controlling height.
                    minHeight: '0',
                    toolbar: (tei || view?.format === 'plain')
                        ? false
                        : ['bold','italic','|','heading','quote','unordered-list','ordered-list','|','preview','side-by-side','fullscreen'],
                    previewClass: ['editor-preview'],
                    ...(tei ? { previewRender: () => '' } : {}),
                    readOnly: view ? !view.editable : false,
                });
                requestAnimationFrame(() => this.easyMDE?.codemirror.refresh());
                // The formatting tools go to the left of the row under the tabs
                // (formatted text only — plain text has none).
                this.moveEditorToolbar();
                this.mountTextZoom();
                if (view.format === 'markdown') this.easyMDE.toggleSideBySide();
                // Autosave status sits at the right of the row for editable text.
                const editor = this.easyMDE;
                const resultsPath = selectedYamlPath;
                const saveState = this.textSave = (view ? view.editable : true) ? {
                    statusEl: null,
                    onSave: () => this.textAutosave?.flush(),
                } : null;
                if (saveState) {
                    this.textAutosave = createTextAutosave({
                        initialValue: editor.value(),
                        read: () => editor.value(),
                        write: (value) => window.pywebview.api.update_display_text(nodeId, value, resultsPath),
                        report: (message, isError) => {
                            if (saveState.statusEl) flashStatus(saveState.statusEl, message, isError);
                            if (message === 'Saved') rerunArtefactSearch();
                            if (isError) console.error('Text autosave failed:', nodeId, message);
                        },
                    });
                    const autosave = this.textAutosave;
                    const changed = () => autosave.changed();
                    editor.codemirror.on('change', changed);
                    this._stopTextAutosave = () => {
                        editor.codemirror.off('change', changed);
                        return autosave.dispose();
                    };
                }
            };

            mountEditor(view.text, isTei);
            if (isTei) this.easyMDE.codemirror.setOption('mode', 'xml');
            if (view.format === 'plain') this.easyMDE.codemirror.setOption('mode', 'text/plain');
            markSearchInEditor(this);
            this.renderPager(); // now that Save may exist
        };

        // The slot at the left of the row under the tabs, for the text-editing tools.
        PanelController.prototype.nodeToolsEl = function() {
            const header = this.el.headerType && this.el.headerType.parentElement;
            return header ? header.querySelector('.node-tools') : null;
        };

        PanelController.prototype.clearNodeTools = function() {
            const slot = this.nodeToolsEl();
            if (slot) slot.replaceChildren();
        };

        // EasyMDE builds its formatting toolbar above the text; move it up into the row
        // under the tabs. The buttons keep working (they refer to the editor, not to
        // where they sit).
        PanelController.prototype.moveEditorToolbar = function() {
            const slot = this.nodeToolsEl();
            if (!slot) return;
            slot.replaceChildren();
            const toolbar = this.el.textDisplay && this.el.textDisplay.querySelector('.editor-toolbar');
            if (toolbar) slot.appendChild(toolbar);
        };

        // A−/A+ text-size buttons after the formatting tools, for every text format.
        PanelController.prototype.mountTextZoom = function() {
            const slot = this.nodeToolsEl();
            if (!slot) return;
            const group = document.createElement('span');
            group.className = 'text-zoom';
            group.setAttribute('role', 'group');
            group.setAttribute('aria-label', 'Text size');
            const button = (label, title, direction) => {
                const element = document.createElement('button');
                element.type = 'button';
                element.textContent = label;
                element.title = title;
                element.setAttribute('aria-label', title);
                element.addEventListener('click', () => {
                    const current = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--text-zoom')) || 1;
                    applyTextZoom(direction ? stepTextZoom(current, direction) : 1);
                });
                return element;
            };
            const level = button('', 'Reset text size', 0);
            level.classList.add('text-zoom-level');
            group.append(button('A−', 'Decrease text size', -1), level, button('A+', 'Increase text size', 1));
            slot.appendChild(group);
            applyTextZoom(parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--text-zoom')) || storedTextZoom());
        };

        PanelController.prototype.stopTextAutosave = function() {
            const stop = this._stopTextAutosave;
            this._stopTextAutosave = null;
            this.textAutosave = null;
            if (stop) this._pendingTextSave = stop();
            return this._pendingTextSave || Promise.resolve();
        };

}

async function renderTextArtefact(controller, view, result) {
    controller.clearPageArtefact();
    await controller.renderTextNodeAsPage(result.id, result.type, view);
}

if (typeof module !== "undefined" && module.exports) {
    module.exports = {installTextDisplay, renderTextArtefact, stepTextZoom, TEXT_ZOOM};
} else {
    registerArtefactRenderer("text", renderTextArtefact);
}
