// startupActions — pure decision logic for what should happen once the
// pywebview API becomes ready. DOM-free so it's unit testable without a
// browser; index.html's pywebviewready handler just carries out whatever
// this returns. Loaded as a plain classic script in index.html (defines a
// global, same as everything else there), and via require() in tests/.
//
// loadWorkflowAssets (theme.css/logo) is unconditional: it only needs the
// live pipeline object, never a selected specimen folder, so it must not be
// gated behind whether initial items were provided at launch.
function startupActions(initialItemsResult) {
    const provided = !!(
        initialItemsResult &&
        initialItemsResult.success &&
        initialItemsResult.provided
    );
    return {
        loadWorkflowAssets: true,
        handleParentFolderResult: provided,
    };
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { startupActions };
}
