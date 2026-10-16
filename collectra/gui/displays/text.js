// Text, Markdown and XML editors, autosave and editor toolbar lifecycle.
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
    module.exports = {installTextDisplay, renderTextArtefact};
} else {
    registerArtefactRenderer("text", renderTextArtefact);
}
