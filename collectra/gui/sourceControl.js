// History and Changes viewer: fixed operations exposed by the Python bridge.
(function () {
    function setupSourceControl(document, api) {
        const viewer = document.getElementById('gitModal');
        const commit = document.getElementById('gitCommitModal');
        const output = document.getElementById('gitOutput');
        const repositoryLabel = document.getElementById('gitRepository');
        const message = document.getElementById('gitCommitMessage');
        const confirm = document.getElementById('gitCommitConfirm');
        const commitError = document.getElementById('gitCommitError');
        const buttons = [...viewer.querySelectorAll('[data-git-action]')];
        let repository = '';
        let busy = false;
        let opener;

        function showRepository(path) {
            repositoryLabel.textContent = `Project folder: ${path}`;
        }

        function showSaveError(text) {
            commitError.textContent = text;
            commitError.hidden = !text;
        }

        function setBusy(value) {
            busy = value;
            buttons.forEach(button => { button.disabled = value; });
            confirm.disabled = value || !message.value.trim();
            viewer.setAttribute('aria-busy', String(value));
            document.getElementById('gitToolbarBtn').disabled = value;
            message.disabled = value;
        }

        async function run(action, commitMessage = '') {
            if (busy) return;
            setBusy(true);
            output.textContent = 'Working…';
            try {
                const result = await api().git_action(action, commitMessage, repository);
                if (result.repository) {
                    repository = result.repository;
                    showRepository(repository);
                } else if (!repository) {
                    showRepository('not found');
                }
                output.textContent = result.success ? result.output : (result.error || 'Something went wrong.');
                if (action === 'save') {
                    showSaveError(result.success ? '' : output.textContent);
                    if (result.success) closeCommit();
                }
            } catch (error) {
                output.textContent = String(error);
                if (action === 'save') showSaveError(String(error));
            } finally {
                setBusy(false);
            }
        }

        function closeCommit() {
            commit.classList.remove('visible');
            buttons.find(button => button.dataset.gitAction === 'save').focus();
        }

        document.getElementById('gitToolbarBtn').addEventListener('click', () => {
            opener = document.activeElement;
            repository = '';
            showRepository('…');
            viewer.classList.add('visible');
            document.getElementById('gitClose').focus();
            run('status');
        });
        document.getElementById('gitClose').addEventListener('click', () => {
            viewer.classList.remove('visible');
            opener?.focus();
        });
        buttons.forEach(button => button.addEventListener('click', () => {
            const action = button.dataset.gitAction;
            if (action === 'save') {
                message.value = '';
                showSaveError('');
                confirm.disabled = true;
                commit.classList.add('visible');
                message.focus();
            } else {
                run(action);
            }
        }));
        message.addEventListener('input', () => { confirm.disabled = busy || !message.value.trim(); });
        confirm.addEventListener('click', () => run('save', message.value));
        document.getElementById('gitCommitCancel').addEventListener('click', closeCommit);
        document.addEventListener('keydown', event => {
            if (event.key !== 'Escape') return;
            if (commit.classList.contains('visible')) {
                closeCommit();
                event.preventDefault();
            } else if (viewer.classList.contains('visible')) {
                document.getElementById('gitClose').click();
                event.preventDefault();
            }
        });
    }
    if (typeof module !== 'undefined') module.exports = {setupSourceControl};
    if (typeof window !== 'undefined') {
        window.addEventListener('DOMContentLoaded', () => setupSourceControl(document, () => window.pywebview.api));
    }
})();
