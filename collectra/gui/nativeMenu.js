// Native menu callbacks share the toolbar's actions and current panel focus.
async function runNativeMenuAction(name) {
    if (!apiReady || document.querySelector('.modal-overlay.visible')) return;
    const controller = panelControllers[focusedPanel];
    const click = (id) => document.getElementById(id)?.click();
    const actions = {
        open: selectCollectra,
        reveal: async () => {
            const path = collectraFolders[currentFolderIndex]?.yaml_path || selectedYamlPath;
            if (!path) return;
            const result = await window.pywebview.api.reveal_in_folder(path);
            if (!result.success) throw new Error(result.error);
        },
        'close-tab': () => {
            const tab = controller?.panel.getActiveTab();
            if (tab) return controller.closeTab(tab.id);
        },
        'previous-file': goPrevFolder,
        'next-file': goNextFolder,
        'find-file': () => {
            if (sidebarRegion.collapsed) toggleSidebarCollapse();
            const input = document.getElementById('folderSearchInput');
            input.focus();
            input.select();
        },
        'find-artefacts': openArtefactSearch,
        'previous-match': () => stepArtefactSearch(-1),
        'next-match': () => stepArtefactSearch(1),
        sidebar: toggleSidebarCollapse,
        graph: openDagTab,
        artefacts: () => click('artefactsToolbarBtn'),
        tasks: () => click('tasksToolbarBtn'),
        'fit-graph': resetDag,
        'fit-image': () => controller?.pageViewer?.viewport.goHome(),
        'split-left': () => createPanel('left'),
        'split-right': () => createPanel('right'),
        'split-above': () => createPanel('above'),
        'split-below': () => createPanel('below'),
        'collapse-panel': () => {
            const entry = panelCollapseRegistry[focusedPanel];
            if (!entry || isOnlyExpandedPanel(focusedPanel)) return;
            if (entry.isCollapsed()) entry.expand(); else entry.collapse();
            updatePanelCloseButtons();
        },
        'remove-panel': () => removePanelByKey(focusedPanel),
        'choose-layout': () => click('newPanelToolbarBtn'),
        'save-layout': showSaveLayoutDialog,
        'source-control': () => click('gitToolbarBtn'),
        'check-updates': () => showUpdatesDialog(),
        statistics: openStatsModal,
        'load-details': openInfoModal,
    };
    try {
        await actions[name]?.();
    } catch (error) {
        console.error('Menu action failed:', name, error);
        alert(error.message);
    }
}
