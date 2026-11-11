// What the Run confirmation panel shows for a run preview from the backend
// (get_run_preview): one row per task saying whether it runs, re-runs or is
// skipped, and whether the run can start. With overwrite ticked, every task
// runs again and replaces its results.
const RUN_ROW_WORDS = {run: 'will run', rerun: 'will re-run', skip: 'skipped'};

function runDialogContent(preview, overwrite = false) {
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
    // A page that can run may still carry a warning, such as results made by
    // another version of the pipeline.
    const warnings = [
        canRun ? preview.message : '',
        overwrite ? `${rows.length === 1 ? 'The task' : `All ${rows.length} tasks`} will run again on this image `
            + `and replace ${rows.length === 1 ? 'its' : 'their'} results, including edits and annotations you made.` : '',
        rerun && !overwrite ? 'Re-run tasks replace their existing output, including edits.' : '',
    ];
    return {
        title: 'Run the pipeline on this page?',
        rows,
        warning: warnings.filter(Boolean).join(' '),
        kept: canRun && !rerun ? 'Your edits are kept.' : '',
        canRun,
        runLabel: `Run ${count} ${count === 1 ? 'task' : 'tasks'}`,
        message,
    };
}

function pagesText(count) {
    return `${count} ${count === 1 ? 'page' : 'pages'}`;
}

function agree(count, one, many) {
    return `${count} ${count === 1 ? one : many}`;
}

// What the panel shows for a run of the whole collection
// (get_collection_preview): how many pages have tasks to run, on how many pages
// each task runs, and the pages that cannot run.
function collectionDialogContent(preview, overwrite = false) {
    const {pages = 0, to_replace = 0, first_time = 0, to_run = 0, tasks = [], unplannable = []} = preview || {};
    const planned = to_replace + first_time;
    const cannot = unplannable.length ? `, ${unplannable.length} cannot run` : '';
    const counts = overwrite
        ? `${agree(to_replace, 'has', 'have')} results to replace, ${agree(first_time, 'runs', 'run')} for the first time`
        : `${agree(to_run, 'has', 'have')} tasks to run, ${agree(planned - to_run, 'is', 'are')} complete`;
    const count = overwrite ? planned : to_run;
    let message = '';
    if (!preview || !preview.can_run) {
        message = (preview && preview.message) || 'The pipeline cannot run on this collection.';
    } else if (!count) {
        message = 'Every page that can run already has its results';
    }
    return {
        title: 'Run the whole collection?',
        summary: `${pagesText(pages)}: ${counts}${cannot}`,
        rows: tasks.map((task) => ({
            text: `${task.name} — runs on ${pagesText(task.pages)}`
                + (task.replace ? ` (replaces results on ${pagesText(task.replace)})` : ''),
        })),
        unplannable: unplannable.map((page) => ({text: `${page.name} — cannot run (${page.reason})`})),
        warning: overwrite ? `All tasks will run again on ${pagesText(planned)} and replace their results, `
            + 'including edits and annotations you made.' : 'Nothing existing is replaced.',
        canRun: !message,
        runLabel: `Run ${pagesText(count)}`,
        message,
    };
}

// The progress line for a get_run_status result. The backend may not know
// which task is running (current is null), so only the counts are promised.
function runStatusText(status, stopping = false) {
    if (!status) return '';
    if (status.running) {
        const unit = status.unit === 'pages' ? ' pages' : '';
        const counts = `${status.done} of ${status.total}${unit}`;
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
// was open when the run started also counts. A run of the collection reports
// its folders instead of one folder.
function runIsForOpenPage(status, openYamlPath, startedYamlPath = '') {
    if (!status || !openYamlPath) return false;
    if (startedYamlPath && openYamlPath === startedYamlPath) return true;
    const trim = (path) => String(path || '').replace(/[\\/]+$/, '');
    const parent = trim(openYamlPath).replace(/[\\/][^\\/]*$/, '');
    return (status.folders || [status.folder]).map(trim).some((folder) => folder && folder === parent);
}

if (typeof module !== 'undefined') module.exports = {runDialogContent, collectionDialogContent, runStatusText, runIsForOpenPage};
