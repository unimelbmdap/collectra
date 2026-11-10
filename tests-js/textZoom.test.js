const {stepTextZoom, TEXT_ZOOM} = require('../collectra/gui/displays/text.js');
const source = require('./displaySources.js');

it('steps text size in tenths and stops at the limits', () => {
    expect(stepTextZoom(1, 1)).toBe(1.1);
    expect(stepTextZoom(1, -1)).toBe(0.9);
    // No floating-point drift after repeated steps.
    let zoom = 1;
    for (let i = 0; i < 3; i++) zoom = stepTextZoom(zoom, 1);
    expect(zoom).toBe(1.3);
    expect(stepTextZoom(TEXT_ZOOM.max, 1)).toBe(TEXT_ZOOM.max);
    expect(stepTextZoom(TEXT_ZOOM.min, -1)).toBe(TEXT_ZOOM.min);
});

it('mounts the controls with the editor and scales source and preview', () => {
    expect(source).toMatch(/this\.moveEditorToolbar\(\);\s*this\.mountTextZoom\(\);/);
    expect(source).toMatch(/\.page-text-display \.editor-preview,[\s\S]*?font-size: calc\(var\(--text-zoom, 1\) \* 1em\)/);
    // Must outrank EasyMDE's later ".EasyMDEContainer .CodeMirror { font: inherit }".
    expect(source).toMatch(/\.page-text-display \.EasyMDEContainer \.CodeMirror,\s*\.page-text-display \.editor-preview,/);
    expect(source).toMatch(/\.cm-header-1 \{ font-size: calc\(var\(--text-zoom, 1\)/);
});
