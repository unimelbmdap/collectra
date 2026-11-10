const {runDialogContent, runStatusText, runIsForOpenPage} = require('../collectra/gui/runDialog.js');

const task = (name, action, reason) => ({name, action, reason});
const preview = (tasks, extra = {}) => ({success: true, can_run: true, tasks, message: '', ...extra});

describe('runDialogContent (the run confirmation panel)', () => {
    it('asks whether to run the pipeline on this page', () => {
        expect(runDialogContent(preview([task('ocr', 'run', 'no output yet')])).title)
            .toBe('Run the pipeline on this page?');
    });

    it('says what happens to each task, in the given order', () => {
        const content = runDialogContent(preview([
            task('ocr', 'run', 'no output yet'),
            task('classify', 'rerun', 'upstream ocr runs'),
            task('detect', 'skip', 'already has output'),
        ]));
        expect(content.rows).toEqual([
            {kind: 'run', text: 'ocr — will run (no output yet)'},
            {kind: 'rerun', text: 'classify — will re-run (upstream ocr runs)'},
            {kind: 'skip', text: 'detect — skipped (already has output)'},
        ]);
    });

    it('labels the Run button with how many tasks run or re-run, singular and plural', () => {
        const one = runDialogContent(preview([task('ocr', 'run', 'no output yet'), task('d', 'skip', 'already has output')]));
        const two = runDialogContent(preview([task('ocr', 'run', 'no output yet'), task('c', 'rerun', 'upstream ocr runs')]));
        expect(one.runLabel).toBe('Run 1 task');
        expect(two.runLabel).toBe('Run 2 tasks');
        expect(one.canRun).toBe(true);
        expect(one.message).toBe('');
    });

    it('warns that re-run tasks replace their output only when a task re-runs', () => {
        const rerun = runDialogContent(preview([task('ocr', 'run', 'no output yet'), task('c', 'rerun', 'upstream ocr runs')]));
        expect(rerun.warning).toBe('Re-run tasks replace their existing output, including edits.');
        expect(rerun.kept).toBe('');
    });

    it('says edits are kept when no task re-runs', () => {
        const plain = runDialogContent(preview([task('ocr', 'run', 'no output yet'), task('d', 'skip', 'already has output')]));
        expect(plain.warning).toBe('');
        expect(plain.kept).toBe('Your edits are kept.');
    });

    it('cannot run, and shows the backend message, when the backend says it cannot', () => {
        const content = runDialogContent({success: true, can_run: false, tasks: [], message: 'Open a page first'});
        expect(content.canRun).toBe(false);
        expect(content.message).toBe('Open a page first');
        expect(content.rows).toEqual([]);
        expect(content.kept).toBe('');
    });

    it('cannot run when every task already has output', () => {
        const content = runDialogContent(preview([task('d', 'skip', 'already has output')]));
        expect(content.canRun).toBe(false);
        expect(content.message).toBe('Every task already has output; there is nothing to run.');
        expect(content.kept).toBe('');
    });

    it('cannot run when there is no preview at all', () => {
        for (const missing of [null, undefined, {}]) {
            const content = runDialogContent(missing);
            expect(content.canRun).toBe(false);
            expect(content.message).toBe('The pipeline cannot run on this page.');
        }
    });
});

describe('runStatusText (the progress line)', () => {
    const status = (extra) => ({running: true, finished: false, folder: '/p', current: null,
        done: 0, total: 2, summary: null, error: null, ...extra});

    it('counts finished tasks without naming one when the backend does not say which runs', () => {
        expect(runStatusText(status({done: 1}))).toBe('Running… (1 of 2 done)');
        expect(runStatusText(status({current: ''}))).toBe('Running… (0 of 2 done)');
    });

    it('names the running task when the backend says which it is', () => {
        expect(runStatusText(status({current: 'ocr'}))).toBe('Running ocr (1 of 2)…');
    });

    it('says it is stopping once Stop was pressed', () => {
        expect(runStatusText(status({done: 1}), true)).toBe('Stopping after the current task… (1 of 2 done)');
    });

    it('gives the summary and points to the log when finished', () => {
        expect(runStatusText(status({running: false, finished: true, done: 2, summary: '2 ran, 2 skipped, 0 failed'})))
            .toBe('Run finished: 2 ran, 2 skipped, 0 failed. See run-log.txt.');
    });

    it('adds the error after the summary', () => {
        expect(runStatusText(status({running: false, finished: true, summary: '1 ran, 0 skipped, 0 failed',
            error: 'The page could not be reloaded after the run: bad yaml'})))
            .toBe('Run finished: 1 ran, 0 skipped, 0 failed. See run-log.txt. The page could not be reloaded after the run: bad yaml');
    });

    it('gives just the error when there is no summary', () => {
        expect(runStatusText(status({running: false, finished: true, error: 'The run could not be recorded: x'})))
            .toBe('Run failed: The run could not be recorded: x');
    });

    it('is empty when there is no run', () => {
        expect(runStatusText(status({running: false, finished: false}))).toBe('');
        expect(runStatusText(null)).toBe('');
    });
});

describe('runIsForOpenPage (reload only the page the run belongs to)', () => {
    it('is true when the open results file is in the run folder', () => {
        expect(runIsForOpenPage({folder: '/data/page1'}, '/data/page1/results.yaml')).toBe(true);
        expect(runIsForOpenPage({folder: 'C:\\data\\page1'}, 'C:\\data\\page1\\results.yaml')).toBe(true);
        expect(runIsForOpenPage({folder: '/data/page1/'}, '/data/page1/results.yaml')).toBe(true);
    });

    it('is false when another page is open, or none', () => {
        expect(runIsForOpenPage({folder: '/data/page1'}, '/data/page2/results.yaml')).toBe(false);
        expect(runIsForOpenPage({folder: '/data/page1'}, '')).toBe(false);
        expect(runIsForOpenPage({folder: ''}, '')).toBe(false);
        expect(runIsForOpenPage(null, '/data/page1/results.yaml')).toBe(false);
    });

    it('is true when the open file is the one that was open when the run started', () => {
        // The backend reports the resolved folder, which can differ from the path the page was opened by.
        expect(runIsForOpenPage({folder: '/private/var/page1'}, '/var/page1/results.yaml', '/var/page1/results.yaml')).toBe(true);
        expect(runIsForOpenPage({folder: '/private/var/page1'}, '/var/page2/results.yaml', '/var/page1/results.yaml')).toBe(false);
    });
});
