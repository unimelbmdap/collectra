const {runDialogContent, collectionDialogContent, runStatusText, runIsForOpenPage} = require('../collectra/gui/runDialog.js');

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

describe('runDialogContent with a warning from the backend', () => {
    const warning = 'Warning: these results were made by the Other pipeline, not demo.';

    it('shows the warning and can still run', () => {
        const content = runDialogContent(preview([task('ocr', 'run', 'no output yet')], {message: warning}));
        expect(content.canRun).toBe(true);
        expect(content.message).toBe('');
        expect(content.warning).toBe(warning);
    });

    it('puts the warning before the re-run warning', () => {
        const content = runDialogContent(preview([task('ocr', 'run', 'no output yet'),
            task('c', 'rerun', 'upstream ocr runs')], {message: warning}));
        expect(content.warning).toBe(`${warning} Re-run tasks replace their existing output, including edits.`);
    });

    it('still says there is nothing to run when every task has output', () => {
        const content = runDialogContent(preview([task('d', 'skip', 'already has output')], {message: warning}));
        expect(content.canRun).toBe(false);
        expect(content.message).toBe('Every task already has output; there is nothing to run.');
    });
});

describe('runDialogContent with "Overwrite existing results" ticked', () => {
    const ticked = preview([task('ocr', 'rerun', 'overwrite'), task('classify', 'rerun', 'overwrite'),
        task('detect', 'run', 'no output yet')]);

    it('warns that every task runs again and replaces its results', () => {
        const content = runDialogContent(ticked, true);
        expect(content.warning).toBe('All 3 tasks will run again on this image and replace their results, '
            + 'including edits and annotations you made.');
        expect(content.kept).toBe('');
        expect(content.rows.map((row) => row.text)).toEqual(['ocr — will re-run (overwrite)',
            'classify — will re-run (overwrite)', 'detect — will run (no output yet)']);
        expect(content.runLabel).toBe('Run 3 tasks');
    });

    it('keeps the backend warning first', () => {
        const content = runDialogContent({...ticked, message: 'Warning: old.'}, true);
        expect(content.warning).toMatch(/^Warning: old\. All 3 tasks will run again/);
    });

    it('says "the task" for a one-task pipeline', () => {
        expect(runDialogContent(preview([task('ocr', 'rerun', 'overwrite')]), true).warning)
            .toBe('The task will run again on this image and replace its results, including edits and annotations you made.');
    });

    it('is unchanged when unticked', () => {
        expect(runDialogContent(ticked, false)).toEqual(runDialogContent(ticked));
        expect(runDialogContent(ticked).warning).toBe('Re-run tasks replace their existing output, including edits.');
    });
});

describe('collectionDialogContent (the panel for the whole collection)', () => {
    const collection = (extra = {}) => ({success: true, can_run: true, pages: 32, to_replace: 12, first_time: 20, to_run: 5,
        tasks: [{name: 'ocr', pages: 20, replace: 0}, {name: 'classify', pages: 1, replace: 1}],
        unplannable: [], message: '', ...extra});

    it('asks whether to run the whole collection and sums up its pages', () => {
        const content = collectionDialogContent(collection());
        expect(content.title).toBe('Run the whole collection?');
        expect(content.summary).toBe('32 pages: 5 have tasks to run, 27 are complete');
        expect(content.canRun).toBe(true);
        expect(content.message).toBe('');
        expect(content.runLabel).toBe('Run 5 pages');
    });

    it('counts the pages with results to replace and every planned page when ticked', () => {
        const content = collectionDialogContent(collection(), true);
        expect(content.summary).toBe('32 pages: 12 have results to replace, 20 run for the first time');
        expect(content.runLabel).toBe('Run 32 pages');
        expect(collectionDialogContent(collection({to_replace: 1, first_time: 0, to_run: 0}), true).runLabel)
            .toBe('Run 1 page');
    });

    it('cannot run when no page has a task to run', () => {
        const content = collectionDialogContent(collection({to_run: 0}));
        expect(content.summary).toBe('32 pages: 0 have tasks to run, 32 are complete');
        expect(content.canRun).toBe(false);
        expect(content.message).toBe('Every page that can run already has its results');
    });

    it('counts the pages that cannot run and lists them with the reason', () => {
        const content = collectionDialogContent(collection({pages: 2, to_replace: 0, first_time: 1, to_run: 1,
            unplannable: [{name: 'p9', reason: 'the image copy is missing'}]}));
        expect(content.summary).toBe('2 pages: 1 has tasks to run, 0 are complete, 1 cannot run');
        expect(collectionDialogContent(collection({pages: 2, to_replace: 1, first_time: 0, to_run: 0,
            unplannable: [{name: 'p9', reason: 'x'}]}), true).summary)
            .toBe('2 pages: 1 has results to replace, 0 run for the first time, 1 cannot run');
        expect(content.unplannable).toEqual([{text: 'p9 — cannot run (the image copy is missing)'}]);
    });

    it('leads with the listed pages and makes the verbs agree with one page', () => {
        const one = collection({pages: 1, to_replace: 0, first_time: 1, to_run: 0});
        expect(collectionDialogContent(one).summary).toBe('1 page: 0 have tasks to run, 1 is complete');
        expect(collectionDialogContent(one, true).summary).toBe('1 page: 0 have results to replace, 1 runs for the first time');
        const listed = collection({pages: 4, to_replace: 0, first_time: 1, to_run: 0,
            unplannable: [{name: 'p8', reason: 'x'}, {name: 'p9', reason: 'x'}, {name: 'p10', reason: 'x'}]});
        expect(collectionDialogContent(listed).summary).toBe('4 pages: 0 have tasks to run, 1 is complete, 3 cannot run');
    });

    it('says on how many pages each task runs, and replaces results, singular and plural', () => {
        expect(collectionDialogContent(collection()).rows).toEqual([
            {text: 'ocr — runs on 20 pages'},
            {text: 'classify — runs on 1 page (replaces results on 1 page)'},
        ]);
        expect(collectionDialogContent(collection({tasks: [{name: 'ocr', pages: 3, replace: 2}]})).rows)
            .toEqual([{text: 'ocr — runs on 3 pages (replaces results on 2 pages)'}]);
    });

    it('says nothing existing is replaced when unticked', () => {
        expect(collectionDialogContent(collection()).warning).toBe('Nothing existing is replaced.');
    });

    it('says every task runs again on every page when ticked', () => {
        expect(collectionDialogContent(collection(), true).warning).toBe('All tasks will run again on 32 pages '
            + 'and replace their results, including edits and annotations you made.');
    });

    it('labels the Run button for one page', () => {
        expect(collectionDialogContent(collection({to_run: 1})).runLabel).toBe('Run 1 page');
    });

    it('cannot run, and shows the backend message, when the backend says it cannot', () => {
        const content = collectionDialogContent(collection({can_run: false, message: 'Every page already has its results'}));
        expect(content.canRun).toBe(false);
        expect(content.message).toBe('Every page already has its results');
        const missing = collectionDialogContent({can_run: false, tasks: [], message: 'bridge gone'});
        expect(missing.message).toBe('bridge gone');
        expect(missing.rows).toEqual([]);
        expect(missing.unplannable).toEqual([]);
    });
});

describe('runStatusText for a run of the collection', () => {
    const status = (extra) => ({running: true, finished: false, folder: '', folders: ['/p1', '/p2'], unit: 'pages',
        current: null, done: 3, total: 34, summary: null, error: null, ...extra});

    it('counts pages', () => {
        expect(runStatusText(status())).toBe('Running… (3 of 34 pages done)');
        expect(runStatusText(status(), true)).toBe('Stopping after the current task… (3 of 34 pages done)');
    });

    it('gives the summary unchanged when finished', () => {
        expect(runStatusText(status({running: false, finished: true, summary: '34 pages: 5 ran, 0 skipped, 0 failed'})))
            .toBe('Run finished: 34 pages: 5 ran, 0 skipped, 0 failed. See run-log.txt.');
    });
});

describe('runIsForOpenPage for a run of the collection', () => {
    const status = {folder: '', folders: ['/data/page1', '/data/page2'], unit: 'pages'};

    it('is true when the open page is one of the run folders', () => {
        expect(runIsForOpenPage(status, '/data/page2/results.yaml')).toBe(true);
    });

    it('is false when the open page is not in the run', () => {
        expect(runIsForOpenPage(status, '/data/page3/results.yaml')).toBe(false);
        expect(runIsForOpenPage({...status, folders: []}, '/data/page3/results.yaml')).toBe(false);
    });
});
