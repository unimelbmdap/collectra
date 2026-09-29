const { startupActions } = require('../collectra/gui/startupActions.js');

describe('startupActions', () => {
    it('always loads workflow assets when initial items were provided', () => {
        expect(startupActions({ success: true, provided: true, folders: [] }).loadWorkflowAssets).toBe(true);
    });

    it('always loads workflow assets even when initial items were NOT provided — this is the fix: theme/logo must not depend on a folder being pre-selected at launch', () => {
        expect(startupActions({ success: true, provided: false }).loadWorkflowAssets).toBe(true);
    });

    it('always loads workflow assets even when get_initial_items failed', () => {
        expect(startupActions({ success: false }).loadWorkflowAssets).toBe(true);
    });

    it('handles the parent folder result only when items were actually provided', () => {
        expect(startupActions({ success: true, provided: true }).handleParentFolderResult).toBe(true);
        expect(startupActions({ success: true, provided: false }).handleParentFolderResult).toBe(false);
    });

    it('does not handle the parent folder result when the initial call failed', () => {
        expect(startupActions({ success: false, provided: true }).handleParentFolderResult).toBe(false);
    });
});
