const { matchesFolderSearch, visibleFolderIndices } = require('../collectra/gui/folderSearch.js');

describe('matchesFolderSearch', () => {
    it('matches when the query is a substring (case-insensitive)', () => {
        expect(matchesFolderSearch('MMRIRN1766879_P360969', 'p360969')).toBe(true);
    });

    it('does not match when the query is not a substring', () => {
        expect(matchesFolderSearch('page_01', 'page_02')).toBe(false);
    });

    it('empty query matches everything', () => {
        expect(matchesFolderSearch('anything', '')).toBe(true);
    });

    it('whitespace-only query matches everything', () => {
        expect(matchesFolderSearch('anything', '   ')).toBe(true);
    });

    it('matches "1" but not "2" for page "01"', () => {
        expect(matchesFolderSearch('page_01', '1')).toBe(true);
        expect(matchesFolderSearch('page_01', '2')).toBe(false);
    });

    it('matches "0" for both "page_01" and "page_02"', () => {
        expect(matchesFolderSearch('page_01', '0')).toBe(true);
        expect(matchesFolderSearch('page_02', '0')).toBe(true);
    });
});

describe('visibleFolderIndices', () => {
    const folders = [{ name: 'page_01' }, { name: 'page_02' }, { name: 'other' }];

    it('returns every index for an empty query — this is what backs "reset search on a new project load": once the query is cleared, every folder in the freshly-loaded list must be visible again', () => {
        expect(visibleFolderIndices(folders, '')).toEqual([0, 1, 2]);
    });

    it('returns only the indices whose name matches the query', () => {
        expect(visibleFolderIndices(folders, '01')).toEqual([0]);
    });

    it('returns indices for every match when the query matches more than one', () => {
        expect(visibleFolderIndices(folders, '0')).toEqual([0, 1]);
    });

    it('returns an empty array when nothing matches', () => {
        expect(visibleFolderIndices(folders, 'zzz')).toEqual([]);
    });

    it('returns an empty array for an empty folder list regardless of query', () => {
        expect(visibleFolderIndices([], 'anything')).toEqual([]);
    });
});
