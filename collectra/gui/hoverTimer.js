// Hover cards (tab, DAG, image box) hide themselves after a few seconds, even
// if the pointer never leaves — e.g. because the element under it was redrawn.
const HOVER_HIDE_MS = 4000;
const hoverTimers = new WeakMap();

function cancelHoverHide(card) {
    if (!card) return;
    clearTimeout(hoverTimers.get(card));
    hoverTimers.delete(card);
}

// (Re)start the countdown for a card that has just been shown.
function scheduleHoverHide(card) {
    if (!card) return;
    cancelHoverHide(card);
    hoverTimers.set(card, setTimeout(() => {
        card.style.display = 'none';
        hoverTimers.delete(card);
    }, HOVER_HIDE_MS));
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {HOVER_HIDE_MS, scheduleHoverHide, cancelHoverHide};
}
