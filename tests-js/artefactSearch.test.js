const {
    matchingLabels, stepIndex, searchSummary, findRanges, searchTargetPanel, isSearchShortcut,
} = require('../collectra/gui/artefactSearch.js');

const match = (id, label, field = 'text') => ({id, label, type: 'collectra.Text', field, snippet: ''});
const matches = [match('a1', 'alpha'), match('b1', 'beta'), match('b2', 'beta'), match('c1', 'gamma')];

describe('matchingLabels', () => {
    it('is the set of DAG labels that have at least one match', () => {
        expect([...matchingLabels(matches)].sort()).toEqual(['alpha', 'beta', 'gamma']);
    });

    it('is empty with no matches', () => {
        expect(matchingLabels([]).size).toBe(0);
    });
});

describe('stepIndex', () => {
    it('starts on the first match going forward and the last going back', () => {
        expect(stepIndex(4, -1, 1)).toBe(0);
        expect(stepIndex(4, -1, -1)).toBe(3);
    });

    it('moves one at a time and wraps around at either end', () => {
        expect(stepIndex(4, 0, 1)).toBe(1);
        expect(stepIndex(4, 3, 1)).toBe(0);
        expect(stepIndex(4, 0, -1)).toBe(3);
    });

    it('has nowhere to go without matches', () => {
        expect(stepIndex(0, -1, 1)).toBe(-1);
    });
});

describe('searchSummary with no query', () => {
    it('shows nothing for an empty query', () => {
        expect(searchSummary('', 0, -1)).toBe('');
        expect(searchSummary('   ', 3, 0)).toBe('');
    });
});

describe('findRanges', () => {
    it('finds every occurrence ignoring case', () => {
        expect(findRanges('Fox and fox and FOX', 'fox')).toEqual([
            {from: 0, to: 3}, {from: 8, to: 11}, {from: 16, to: 19},
        ]);
    });

    it('does not overlap matches', () => {
        expect(findRanges('aaaa', 'aa')).toEqual([{from: 0, to: 2}, {from: 2, to: 4}]);
    });

    it('treats the query literally, not as a pattern', () => {
        expect(findRanges('a.c abc', 'a.c')).toEqual([{from: 0, to: 3}]);
    });

    it('returns nothing for an empty query or text', () => {
        expect(findRanges('abc', '')).toEqual([]);
        expect(findRanges('abc', '  ')).toEqual([]);
        expect(findRanges('', 'a')).toEqual([]);
        expect(findRanges(null, 'a')).toEqual([]);
    });
});

describe('searchSummary in the find-bar wording', () => {
    it('reads "17 of 173 matches" while stepping', () => {
        expect(searchSummary('x', 173, 16)).toBe('17 of 173 matches');
        expect(searchSummary('x', 1, 0)).toBe('1 of 1 match');
    });

    it('reads "173 matches" before any is selected', () => {
        expect(searchSummary('x', 173, -1)).toBe('173 matches');
        expect(searchSummary('x', 1, -1)).toBe('1 match');
    });

    it('says "No matches" when there are none', () => {
        expect(searchSummary('x', 0, -1)).toBe('No matches');
    });
});

describe('searchTargetPanel', () => {
    const visible = () => true;
    const at = (map) => (key) => map[key] || {top: 0, left: 0};

    it('is the panel you chose, when it exists and can be seen', () => {
        expect(searchTargetPanel(['top', 'bottom', 'panel2'], 'panel2', visible, at({}))).toBe('panel2');
    });

    it('is the topmost panel when you have not chosen one', () => {
        const where = at({bottom: {top: 400, left: 0}, top: {top: 0, left: 0}, panel2: {top: 100, left: 0}});
        expect(searchTargetPanel(['bottom', 'panel2', 'top'], null, visible, where)).toBe('top');
        expect(searchTargetPanel(['bottom', 'panel2'], undefined, visible, where)).toBe('panel2');
    });

    it('among panels at the same height, the leftmost', () => {
        const where = at({a: {top: 50, left: 600}, b: {top: 50, left: 10}, c: {top: 50, left: 300}});
        expect(searchTargetPanel(['a', 'b', 'c'], null, visible, where)).toBe('b');
    });

    it('is the topmost visible panel when the chosen one was deleted', () => {
        const where = at({bottom: {top: 400, left: 0}, panel3: {top: 0, left: 0}});
        expect(searchTargetPanel(['bottom', 'panel3'], 'panel2', visible, where)).toBe('panel3');
    });

    it('skips panels that cannot be seen, including a chosen one that is collapsed or empty', () => {
        const where = at({top: {top: 0, left: 0}, bottom: {top: 300, left: 0}});
        const notTop = (key) => key !== 'top';
        expect(searchTargetPanel(['top', 'bottom'], 'top', notTop, where)).toBe('bottom');
        expect(searchTargetPanel(['top', 'bottom'], null, notTop, where)).toBe('bottom');
    });

    it('falls back to the first panel when none can be seen, and to null when there are none', () => {
        expect(searchTargetPanel(['top', 'bottom'], 'gone', () => false, at({}))).toBe('top');
        expect(searchTargetPanel([], 'bottom', visible, at({}))).toBe(null);
    });
});

describe('isSearchShortcut', () => {
    const key = (extra) => ({ key: 'f', metaKey: false, ctrlKey: false, shiftKey: false, altKey: false, ...extra });

    it('is Cmd+F or Ctrl+F, whatever the letter case', () => {
        expect(isSearchShortcut(key({ metaKey: true }))).toBe(true);
        expect(isSearchShortcut(key({ ctrlKey: true, key: 'F' }))).toBe(true);
    });

    it('is not plain F, another key, or F with Shift or Alt added', () => {
        expect(isSearchShortcut(key())).toBe(false);
        expect(isSearchShortcut(key({ metaKey: true, key: 'g' }))).toBe(false);
        expect(isSearchShortcut(key({ metaKey: true, shiftKey: true }))).toBe(false);
        expect(isSearchShortcut(key({ ctrlKey: true, altKey: true }))).toBe(false);
    });

    it('copes with an event that has no key', () => {
        expect(isSearchShortcut({ metaKey: true })).toBe(false);
        expect(isSearchShortcut({ metaKey: true, key: undefined })).toBe(false);
    });
});

describe('searchSummary when the search could not run', () => {
    it('shows the problem instead of "No matches"', () => {
        expect(searchSummary('loc', 0, -1, 'Open a page to search')).toBe('Open a page to search');
    });

    it('shows nothing for an empty query even if there was a problem', () => {
        expect(searchSummary('', 0, -1, 'Open a page to search')).toBe('');
    });

    it('is unchanged when there is no problem', () => {
        expect(searchSummary('x', 0, -1, '')).toBe('No matches');
        expect(searchSummary('x', 3, 0)).toBe('1 of 3 matches');
    });
});
