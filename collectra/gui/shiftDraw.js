// Temporarily arm the existing drawing tool before a Shift-drag reaches it.
function mountShiftDraw({element, document, window, canDraw, isDrawing, setDrawing, sync}) {
    let active = false;
    let previous = false;
    let pointer = null;
    let timer;
    let hovered = false;
    function finish(cancel = false) {
        clearTimeout(timer);
        if (!active) return;
        active = false;
        pointer = null;
        setDrawing(cancel ? false : previous);
        sync();
    }
    function arm() {
        if (active || !canDraw()) return;
        previous = isDrawing();
        active = true;
        setDrawing(true);
        sync();
    }
    function enter(event) {
        hovered = true;
        if (event.shiftKey) arm();
    }
    function leave() {
        hovered = false;
        if (pointer === null) finish();
    }
    function down(event) {
        if (event.button !== 0 || !event.shiftKey || !canDraw()) return;
        if (event.pointerType && event.pointerType !== 'mouse' && event.pointerType !== 'pen') return;
        if (event.target.closest?.('button, input, textarea, select, a, [role="button"]')) return;
        arm();
        if (active) pointer = event.pointerId;
    }
    function up(event) {
        if (!active || event.pointerId !== pointer) return;
        // Let Annotorious finish its pointerup/createAnnotation handlers first.
        timer = setTimeout(() => finish(), 0);
    }
    function cancel(event) {
        if (!event || event.pointerId === pointer) finish(true);
    }
    function key(event) {
        if (event.key === 'Escape') finish(true);
        else if (event.key === 'Shift' && hovered
            && !document.activeElement?.closest?.('input, textarea, select, [contenteditable="true"]')) arm();
    }
    const keyup = event => { if (event.key === 'Shift' && pointer === null) finish(); };
    element.addEventListener('pointerenter', enter);
    element.addEventListener('pointerleave', leave);
    element.addEventListener('pointerdown', down, true);
    document.addEventListener('keyup', keyup, true);
    document.addEventListener('pointerup', up, true);
    document.addEventListener('pointercancel', cancel, true);
    document.addEventListener('keydown', key, true);
    const blur = () => finish(true);
    window.addEventListener('blur', blur);
    return () => {
        clearTimeout(timer);
        element.removeEventListener('pointerenter', enter);
        element.removeEventListener('pointerleave', leave);
        element.removeEventListener('pointerdown', down, true);
        document.removeEventListener('keyup', keyup, true);
        document.removeEventListener('pointerup', up, true);
        document.removeEventListener('pointercancel', cancel, true);
        document.removeEventListener('keydown', key, true);
        window.removeEventListener('blur', blur);
    };
}
if (typeof module !== 'undefined') module.exports = {mountShiftDraw};
