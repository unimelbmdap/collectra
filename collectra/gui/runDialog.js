// What the Run confirmation panel shows for a run preview from the backend
// (get_run_preview): one row per task saying whether it runs, re-runs or is
// skipped, and whether the run can start.
const RUN_ROW_WORDS = {run: 'will run', rerun: 'will re-run', skip: 'skipped'};

function runDialogContent(preview) {
    const tasks = (preview && preview.tasks) || [];
    const rows = tasks.map((task) => ({
        kind: task.action,
        text: `${task.name} — ${RUN_ROW_WORDS[task.action] || task.action} (${task.reason})`,
    }));
    const count = rows.filter((row) => row.kind !== 'skip').length;
    const rerun = rows.some((row) => row.kind === 'rerun');
    let message = '';
    if (!preview || !preview.can_run) {
        message = (preview && preview.message) || 'The pipeline cannot run on this page.';
    } else if (!count) {
        message = 'Every task already has output; there is nothing to run.';
    }
    const canRun = !message;
    return {
        title: 'Run the pipeline on this page?',
        rows,
        warning: rerun ? 'Re-run tasks replace their existing output, including edits.' : '',
        kept: canRun && !rerun ? 'Your edits are kept.' : '',
        canRun,
        runLabel: `Run ${count} ${count === 1 ? 'task' : 'tasks'}`,
        message,
    };
}

// The progress line for a get_run_status result. The backend may not know
// which task is running (current is null), so only the counts are promised.
function runStatusText(status, stopping = false) {
    if (!status) return '';
    if (status.running) {
        const counts = `${status.done} of ${status.total}`;
        if (stopping) return `Stopping after the current task… (${counts} done)`;
        if (status.current) return `Running ${status.current} (${status.done + 1} of ${status.total})…`;
        return `Running… (${counts} done)`;
    }
    if (!status.finished) return '';
    if (!status.summary) return `Run failed: ${status.error}`;
    const text = `Run finished: ${status.summary}. See run-log.txt.`;
    return status.error ? `${text} ${status.error}` : text;
}

// Whether a finished run belongs to the page open now, so it should be
// reloaded. The backend reports the resolved folder, which can be spelt
// differently from the path the page was opened by, so the results file that
// was open when the run started also counts.
function runIsForOpenPage(status, openYamlPath, startedYamlPath = '') {
    if (!status || !openYamlPath) return false;
    if (startedYamlPath && openYamlPath === startedYamlPath) return true;
    const trim = (path) => String(path || '').replace(/[\\/]+$/, '');
    const folder = trim(status.folder);
    const parent = trim(openYamlPath).replace(/[\\/][^\\/]*$/, '');
    return Boolean(folder) && folder === parent;
}

if (typeof module !== 'undefined') module.exports = {runDialogContent, runStatusText, runIsForOpenPage};
