// Stable, random-looking pastel colours shared by graph nodes and crop labels.
function nodeColor(label) {
    let hash = 2166136261;
    for (const character of String(label || '')) {
        hash = Math.imul(hash ^ character.codePointAt(0), 16777619);
    }
    const hue = (hash >>> 0) % 360;
    return `hsl(${hue}, 65%, 78%)`;
}

function cropAnnotationStyle(annotation, state) {
    const body = annotation.bodies?.[0] || {};
    const color = nodeColor(body.label || body.id?.replace(/-[^-]+$/, '') || '');
    const selected = !!state?.selected;
    const active = selected || !!state?.hovered;
    return {
        fill: color,
        fillOpacity: active ? 0.35 : 0.08,
        stroke: color,
        strokeOpacity: active ? 1 : 0.85,
        strokeWidth: active ? 4 : 1.5,
    };
}

if (typeof module !== 'undefined') module.exports = {nodeColor, cropAnnotationStyle};
