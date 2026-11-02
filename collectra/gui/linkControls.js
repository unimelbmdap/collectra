// Per-image controls for the Link nodes under it. Each Link gets a button that
// shows its name; its menu renames the Link or sets it to Undefined, which
// removes it. The backend calls are injected so the behaviour can be tested
// without a DOM.

// What the menu offers: every known name (the current one always included),
// the current one marked, then Undefined last.
function linkMenuItems(current, names) {
    const all = [...new Set([...names, current])].sort((a, b) => a.localeCompare(b));
    return [...all.map(name => ({name, current: name === current})), {undefined: true}];
}

function createLinkEditor({link, loadNames, rename, remove, onChanged, isCurrent, onState}) {
    let id = link.id;
    let name = link.name;
    let names = [name];
    let open = false;
    let busy = false;
    let error = '';
    const state = () => ({id, name, names, open, busy, error});
    const emit = () => onState(state());

    async function toggle() {
        if (open) {
            open = false;
            emit();
            return;
        }
        error = '';
        try {
            const result = await loadNames();
            if (!isCurrent()) return;
            if (!result.success) throw new Error(result.error || 'Could not load link names');
            names = result.names;
        } catch (failure) {
            if (!isCurrent()) return;
            names = [name];
            error = failure.message || String(failure);
        }
        open = true;
        emit();
    }

    function close() {
        if (!open) return;
        open = false;
        emit();
    }

    // Runs one change at a time. `change` returns true when something changed.
    async function work(change) {
        if (busy) return;
        busy = true;
        error = '';
        emit();
        try {
            if (await change() && isCurrent()) {
                open = false;
                await onChanged();
            }
        } catch (failure) {
            if (isCurrent()) error = failure.message || String(failure);
        } finally {
            busy = false;
            if (isCurrent()) emit();
        }
    }

    async function choose(requested) {
        const next = String(requested ?? '').trim();
        if (!next || busy) return;
        if (next === name) return close();
        await work(async () => {
            const result = await rename(id, next);
            if (!isCurrent()) return false;
            if (!result.success) throw new Error(result.error || 'Could not rename the link');
            id = result.id;
            name = next;
            return true;
        });
    }

    async function undefine() {
        await work(() => remove(id, name));
    }

    return {toggle, close, choose, undefine, state};
}

function mountLinkControls({viewer, view, loadNames, rename, remove, onChanged, isCurrent}) {
    const links = view.links || [];
    if (!links.length) return [];
    const bar = document.createElement('div');
    bar.className = 'link-controls';
    bar.setAttribute('role', 'group');
    bar.setAttribute('aria-label', 'Links');
    const wrappers = [];
    const editors = [];
    links.forEach(link => {
        const wrapper = document.createElement('div');
        wrapper.className = 'link-control';
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'link-control-button';
        button.setAttribute('aria-haspopup', 'menu');
        const menu = document.createElement('div');
        menu.className = 'link-control-menu';
        menu.setAttribute('role', 'menu');
        menu.hidden = true;
        const entry = document.createElement('input');
        entry.type = 'text';
        entry.className = 'link-control-entry';
        entry.placeholder = 'New name, then Enter';
        entry.setAttribute('aria-label', 'New link name');
        entry.hidden = true;
        const status = document.createElement('span');
        status.className = 'link-control-status';
        status.setAttribute('role', 'status');
        wrapper.append(button, menu, entry, status);
        wrappers.push(wrapper);

        const editor = createLinkEditor({
            link, loadNames, rename, remove, onChanged, isCurrent,
            onState: state => {
                button.textContent = `${state.name} ▾`;
                button.title = `Link "${state.name}": click to change it`;
                button.setAttribute('aria-expanded', String(state.open));
                button.disabled = state.busy;
                menu.hidden = !state.open;
                entry.hidden = !state.open;
                entry.disabled = state.busy;
                status.textContent = state.error || (state.busy ? 'Working…' : '');
                menu.replaceChildren(...linkMenuItems(state.name, state.names).map(item => {
                    const choice = document.createElement('button');
                    choice.type = 'button';
                    choice.setAttribute('role', 'menuitem');
                    choice.className = 'link-control-item';
                    choice.disabled = state.busy;
                    if (item.undefined) {
                        choice.textContent = 'Undefined';
                        choice.title = 'Remove this link';
                        choice.className += ' undefined';
                        choice.onclick = () => editor.undefine();
                    } else {
                        choice.textContent = item.name;
                        if (item.current) choice.setAttribute('aria-current', 'true');
                        choice.onclick = () => editor.choose(item.name);
                    }
                    return choice;
                }));
            },
        });
        button.onclick = () => {
            editors.forEach(other => { if (other !== editor) other.close(); });
            return editor.toggle();
        };
        entry.onkeydown = event => {
            if (event.key === 'Enter') editor.choose(entry.value);
            else if (event.key === 'Escape') editor.close();
        };
        button.textContent = `${link.name} ▾`;
        editors.push(editor);
    });

    // A click anywhere else closes an open menu.
    const closeOthers = event => {
        if (!isCurrent()) return document.removeEventListener('click', closeOthers);
        editors.forEach((editor, index) => {
            if (!wrappers[index].contains?.(event.target)) editor.close();
        });
    };
    document.addEventListener('click', closeOthers);
    for (const eventName of ['pointerdown', 'mousedown', 'touchstart', 'click', 'keydown']) {
        bar.addEventListener(eventName, e => e.stopPropagation());
    }
    bar.append(...wrappers);
    viewer.addControl(bar, {anchor: OpenSeadragon.ControlAnchor.TOP_LEFT});
    return editors;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {linkMenuItems, createLinkEditor, mountLinkControls};
}
