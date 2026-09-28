// folderSearch — pure predicate for the sidebar's page-name search box.
// Case-insensitive substring match; an empty/whitespace-only query matches
// everything. DOM-free so it's unit testable without a browser. Loaded as a
// plain classic script in index.html (defines a global, same as everything
// else there), and via require() in tests/.
function matchesFolderSearch(name, query) {
    const trimmed = (query || '').trim().toLowerCase();
    if (!trimmed) return true;
    return (name || '').toLowerCase().includes(trimmed);
}

// Given a folder list (each with a .name) and a search query, returns the
// indices of the folders that should be visible. An empty query returns every
// index — this is the property that "reset search on a new project load"
// relies on: clearing the query before this runs makes the whole list visible.
function visibleFolderIndices(folders, query) {
    const visible = [];
    folders.forEach((folder, index) => {
        if (matchesFolderSearch(folder.name, query)) visible.push(index);
    });
    return visible;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { matchesFolderSearch, visibleFolderIndices };
}
