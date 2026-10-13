// Source control viewer: fixed operations exposed by the Python bridge.
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
            output.textContent = `Running git ${action}…`;
            try {
                const result = await api().git_action(action, commitMessage, repository);
                if (result.repository) {
                    repository = result.repository;
                    repositoryLabel.textContent = repository;
                }
                output.textContent = result.success ? result.output : (result.error || 'Git failed');
                if (action === 'commit') {
                    commitError.textContent = result.success ? '' : output.textContent;
                    if (result.success) closeCommit();
                }
            } catch (error) {
                output.textContent = String(error);
                if (action === 'commit') commitError.textContent = String(error);
            } finally {
                setBusy(false);
            }
        }

        function closeCommit() {
            commit.classList.remove('visible');
            buttons.find(button => button.dataset.gitAction === 'commit').focus();
        }

        document.getElementById('gitToolbarBtn').addEventListener('click', () => {
            opener = document.activeElement;
            repository = '';
            repositoryLabel.textContent = 'Active results repository';
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
            if (action === 'commit') {
                message.value = '';
                commitError.textContent = '';
                confirm.disabled = true;
                commit.classList.add('visible');
                message.focus();
            } else {
                run(action);
            }
        }));
        message.addEventListener('input', () => { confirm.disabled = busy || !message.value.trim(); });
        confirm.addEventListener('click', () => run('commit', message.value));
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
