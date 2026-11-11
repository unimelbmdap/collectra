// Settings dialog: the environment variables stored for this pipeline, kept
// in the user's config folder (or the system keyring) rather than in the
// pipeline. The backend returns the whole list after every change.
(function () {
    const ENV_NAME = /^[A-Za-z_][A-Za-z0-9_]*$/;

    // Why a typed name cannot be used, or '' when it can (or nothing is typed).
    function envNameProblem(name) {
        if (!name || ENV_NAME.test(name)) return '';
        return 'Use letters, digits and _, not starting with a digit.';
    }

    function envStatus(variable) {
        if (variable.missing) return ['Not in the keyring on this computer', 'missing'];
        if (variable.overridden) return ['Overridden by the shell', 'overridden'];
        if (variable.replaces_shell) return ['Overrides the shell', 'replacing'];
        return ['In use', 'used'];
    }

    // One row per stored variable, as the dialog shows it.
    function envRows(listing) {
        return ((listing && listing.variables) || []).map((variable) => {
            const [status, state] = envStatus(variable);
            return {
                name: variable.name,
                value: variable.value,
                storage: variable.private ? 'Keyring' : 'File',
                status,
                state,
                override: Boolean(variable.override),
            };
        });
    }

    function setupSettings(document, api) {
        const get = (id) => document.getElementById(id);
        const modal = get('settingsModal');
        const table = get('envTable');
        const body = get('envRows');
        const empty = get('envEmpty');
        const path = get('envPath');
        const form = get('envAddForm');
        const name = get('envNameInput');
        const value = get('envValueInput');
        const secret = get('envPrivateInput');
        const secretNote = get('envPrivateNote');
        const override = get('envOverrideInput');
        const add = get('envAddBtn');
        const error = get('envError');
        const clear = get('envClearBtn');
        const confirm = get('envClearConfirm');
        const question = get('envClearQuestion');
        let opener;
        let count = 0;

        function showError(text) {
            error.textContent = text || '';
            error.hidden = !text;
        }

        function cell(text, className = '') {
            const td = document.createElement('td');
            td.textContent = text;
            if (className) td.className = className;
            return td;
        }

        function rowElement(row) {
            const tr = document.createElement('tr');
            tr.className = `env-row env-${row.state}`;
            const overrideCell = document.createElement('td');
            const box = document.createElement('input');
            box.type = 'checkbox';
            box.checked = row.override;
            box.dataset.name = row.name;
            box.setAttribute('aria-label', `${row.name} overrides the shell`);
            box.addEventListener('change', () => call('set_env_override', row.name, box.checked));
            overrideCell.append(box);
            const deleteCell = document.createElement('td');
            const remove = document.createElement('button');
            remove.type = 'button';
            remove.className = 'env-delete';
            remove.textContent = '×';
            remove.dataset.name = row.name;
            remove.setAttribute('aria-label', `Delete ${row.name}`);
            remove.addEventListener('click', () => call('delete_env_variable', row.name));
            deleteCell.append(remove);
            tr.append(
                cell(row.name, 'env-name'),
                cell(row.value, 'env-value'),
                cell(row.storage),
                cell(row.status, 'env-status'),
                overrideCell,
                deleteCell,
            );
            return tr;
        }

        function render(listing) {
            showError(listing.success === false ? listing.error : '');
            const rows = envRows(listing);
            count = rows.length;
            body.replaceChildren(...rows.map(rowElement));
            table.hidden = !count;
            empty.hidden = Boolean(count);
            clear.disabled = !count;
            path.textContent = listing.path ? `Stored for ${listing.pipeline} in ${listing.path}` : '';
            const keyring = listing.keyring_available !== false;
            secret.disabled = !keyring;
            if (!keyring) secret.checked = false;
            secretNote.hidden = keyring;
            syncPrivate();
        }

        async function call(method, ...args) {
            try {
                const listing = await api()[method](...args);
                render(listing);
                return listing;
            } catch (problem) {
                showError(String(problem));
                return null;
            }
        }

        function syncPrivate() {
            value.type = secret.checked ? 'password' : 'text';
        }

        function canAdd() {
            const typed = name.value.trim();
            return Boolean(typed && value.value) && !envNameProblem(typed);
        }

        function updateAdd() {
            showError(envNameProblem(name.value.trim()));
            add.disabled = !canAdd();
        }

        function hideConfirm() {
            confirm.hidden = true;
        }

        function close() {
            hideConfirm();
            modal.classList.remove('visible');
            opener?.focus();
        }

        get('settingsToolbarBtn').addEventListener('click', () => {
            opener = document.activeElement;
            hideConfirm();
            add.disabled = true;
            modal.classList.add('visible');
            name.focus();
            call('get_env_variables');
        });
        get('settingsClose').addEventListener('click', close);
        name.addEventListener('input', updateAdd);
        value.addEventListener('input', updateAdd);
        secret.addEventListener('change', syncPrivate);
        form.addEventListener('submit', async (event) => {
            event.preventDefault();
            if (!canAdd()) return;
            const listing = await call(
                'set_env_variable', name.value.trim(), value.value, secret.checked, override.checked,
            );
            if (!listing || listing.success === false) return;
            name.value = '';
            value.value = '';
            secret.checked = false;
            override.checked = false;
            syncPrivate();
            add.disabled = true;
            name.focus();
        });
        clear.addEventListener('click', () => {
            question.textContent = count === 1 ? 'Delete 1 variable?' : `Delete all ${count} variables?`;
            confirm.hidden = false;
        });
        get('envClearNo').addEventListener('click', hideConfirm);
        get('envClearYes').addEventListener('click', async () => {
            hideConfirm();
            await call('clear_env_variables');
        });
        document.addEventListener('keydown', (event) => {
            if (event.key !== 'Escape' || !modal.classList.contains('visible')) return;
            if (!confirm.hidden) hideConfirm(); else close();
            event.preventDefault();
        });
    }

    if (typeof module !== 'undefined') module.exports = {envRows, envNameProblem, setupSettings};
    if (typeof window !== 'undefined') {
        window.addEventListener('DOMContentLoaded', () => setupSettings(document, () => window.pywebview.api));
    }
})();
