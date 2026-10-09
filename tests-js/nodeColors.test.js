const {nodeColor, cropAnnotationStyle} = require('../collectra/gui/nodeColors.js');

it('keeps colours stable and distinguishes labels with the same prefix', () => {
    expect(nodeColor('main_text_image')).toBe(nodeColor('main_text_image'));
    expect(nodeColor('main_text_image')).not.toBe(nodeColor('main_title_image'));
    expect(nodeColor('apparatus_image')).toMatch(/^hsl\(\d+, 65%, 78%\)$/);
});

it('uses the graph label for every crop of that type', () => {
    const first = cropAnnotationStyle({bodies: [{id: 'anything-001', label: 'main_text_image'}]}, {});
    const second = cropAnnotationStyle({bodies: [{id: 'anything-002', label: 'main_text_image'}]}, {selected: true});
    expect(first.stroke).toBe(nodeColor('main_text_image'));
    expect(second.stroke).toBe(first.stroke);
    expect(second.strokeWidth).toBeGreaterThan(first.strokeWidth);
    expect(cropAnnotationStyle({bodies: [{label: 'apparatus_image'}]}, {}).stroke).not.toBe(first.stroke);
});
