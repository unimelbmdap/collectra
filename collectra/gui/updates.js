async function showUpdatesDialog() {
    const overlay = document.createElement('div');
    overlay.className = 'modal-overlay visible';
    overlay.setAttribute('role', 'dialog');
    overlay.setAttribute('aria-modal', 'true');
    overlay.setAttribute('aria-label', 'Check for updates');
    overlay.innerHTML = `<div class="modal-popup">
        <h3>Check for updates</h3>
        <div class="update-results" role="status">Checking pipeline and Collectra repositories…</div>
        <div class="modal-buttons"><button type="button">Close</button></div>
    </div>`;
    const results = overlay.querySelector('.update-results');
    const close = overlay.querySelector('button');
    const opener = document.activeElement;
    const dismiss = () => { overlay.remove(); opener?.focus(); };
    close.onclick = dismiss;
    overlay.onkeydown = (event) => {
        if (event.key === 'Escape') { event.stopPropagation(); dismiss(); }
    };
    document.body.appendChild(overlay);
    close.focus();
    let repositories = [];
    let busy = false;
    let restartRequired = false;
    const flushEdits = () => Promise.all(Object.values(panelControllers).map((controller) =>
        controller.textAutosave?.flush() || controller._pendingTextSave));

    function render() {
        results.replaceChildren();
        for (const repository of repositories) {
            const section = document.createElement('section');
            const title = document.createElement('h4');
            title.textContent = repository.label;
            const path = document.createElement('p');
            path.textContent = repository.path;
            path.style.overflowWrap = 'anywhere';
            const message = document.createElement('p');
            message.textContent = repository.message;
            section.append(title, path, message);
            if (repository.can_update) {
                const update = document.createElement('button');
                update.textContent = 'Update ' + repository.label;
                update.disabled = busy;
                update.onclick = async () => {
                    if (busy) return;
                    busy = true;
                    repository.message = 'Updating…';
                    render();
                    try {
                        await flushEdits();
                        const result = await window.pywebview.api.install_update(repository.id);
                        if (!result.success) throw new Error(result.error);
                        repositories = repositories.map((item) => item.id === repository.id ? result.repository : item);
                        if (result.repository.status === 'updated') restartRequired = true;
                    } catch (error) {
                        repository.message = 'Update failed: ' + error.message;
                    } finally {
                        busy = false;
                        render();
                    }
                };
                section.appendChild(update);
            }
            results.appendChild(section);
        }
        if (restartRequired) {
            const note = document.createElement('p');
            note.textContent = 'Restart the app to use the updates. Your open results and layout will be restored.';
            const restart = document.createElement('button');
            restart.textContent = 'Restart Now';
            restart.disabled = busy;
            restart.onclick = async () => {
                if (busy) return;
                busy = true;
                restart.disabled = true;
                restart.textContent = 'Restarting…';
                try {
                    const savedEdits = await flushEdits();
                    if (savedEdits.some((saved) => saved === false)) throw new Error('Some edits could not be saved. Resolve the save error before restarting.');
                    const layout = await window.pywebview.api.save_arrangement(currentArrangement());
                    if (!layout.success) throw new Error(layout.error || 'Could not save layout.');
                    const result = await window.pywebview.api.restart_gui();
                    if (!result.success) throw new Error(result.error);
                } catch (error) {
                    note.textContent = 'Could not restart: ' + error.message;
                    busy = false;
                    restart.disabled = false;
                    restart.textContent = 'Restart Now';
                }
            };
            results.append(note, restart);
        }
    }

    try {
        await flushEdits();
        const result = await window.pywebview.api.check_updates();
        if (!result.success) throw new Error(result.error);
        repositories = result.repositories;
        restartRequired = !!result.restart_required;
        render();
    } catch (error) {
        results.textContent = 'Could not check for updates: ' + error.message;
    }
}
