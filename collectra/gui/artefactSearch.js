// Pure logic for the toolbar search: what the matches mean for the DAG, the
// tabs and the stepping controls. The page decides when to run it.

// DAG labels that have at least one matching artefact on the page.
function matchingLabels(matches) {
    return new Set(matches.map((m) => m.label));
}

// The next (direction 1) or previous (-1) match, wrapping around. current is
// -1 before any match has been selected.
function stepIndex(count, current, direction) {
    if (count === 0) return -1;
    if (current < 0) return direction > 0 ? 0 : count - 1;
    return (current + direction + count) % count;
}

// The count shown inside the box, in the find-bar wording: "17 of 173
// matches" once stepping, "173 matches" before.
function searchSummary(query, count, current, problem) {
    if (!(query || '').trim()) return '';
    if (problem) return problem; // the search could not run: say why, not "No matches"
    if (count === 0) return 'No matches';
    const noun = count === 1 ? 'match' : 'matches';
    if (current >= 0) return `${current + 1} of ${count} ${noun}`;
    return `${count} ${noun}`;
}

// Every non-overlapping, case-insensitive occurrence of query in text, as
// character ranges — for marking the words inside a shown text artefact.
function findRanges(text, query) {
    const needle = (query || '').trim();
    if (!needle || !text) return [];
    // Matched in the original text, not a lower-cased copy: lower-casing can change a
    // letter's length (e.g. a dotted capital I), which would shift every later position.
    const pattern = new RegExp(needle.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi');
    return [...text.matchAll(pattern)].map((m) => ({from: m.index, to: m.index + m[0].length}));
}

// Which panel a search result opens in: the panel the user chose (chosenKey, null when
// they have not chosen one) if it still exists and can be seen; otherwise the topmost
// panel on screen (then the leftmost) that can be seen. Falls back to the first panel
// when none can be seen, and to null when there are no panels. positionOf(key) gives
// {top, left}.
function searchTargetPanel(keys, chosenKey, isVisible, positionOf) {
    if (keys.length === 0) return null;
    if (keys.includes(chosenKey) && isVisible(chosenKey)) return chosenKey;
    const visible = keys.filter((key) => isVisible(key));
    if (visible.length === 0) return keys[0];
    return visible.reduce((best, key) => {
        const a = positionOf(key);
        const b = positionOf(best);
        return a.top < b.top || (a.top === b.top && a.left < b.left) ? key : best;
    });
}

// Cmd+F (Mac) or Ctrl+F opens the search — not with Shift or Alt added.
function isSearchShortcut(e) {
    return (e.metaKey || e.ctrlKey) && !e.shiftKey && !e.altKey &&
        typeof e.key === 'string' && e.key.toLowerCase() === 'f';
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {matchingLabels, stepIndex, searchSummary, findRanges, searchTargetPanel, isSearchShortcut};
}
