const fs = require('node:fs');
const html = fs.readFileSync(require.resolve('../collectra/gui/index.html'), 'utf8');
const dialog = html.slice(html.indexOf('id="gitModal"'), html.indexOf('<!-- New-box Popup Modal -->'));
const text = (fragment) => fragment.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ');

describe('History and Changes dialog', () => {
    it('is titled in plain words', () => {
        expect(dialog).toContain('<h3 id="gitTitle">History and Changes</h3>');
        expect(dialog).toContain('aria-label="Close history and changes"');
    });

    it('has four buttons: What\'s changed, Get updates, Save changes, Share my changes', () => {
        const buttons = [...dialog.matchAll(/<button class="btn-cancel" data-git-action="(\w+)">([^<]*)<\/button>/g)]
            .map((m) => [m[1], m[2]]);
        expect(buttons).toEqual([
            ['status', "What's changed"],
            ['pull', 'Get updates'],
            ['save', 'Save changes'],
            ['push', 'Share my changes'],
        ]);
    });

    it('the save popup says what is included, asks for a message and marks it required', () => {
        const popup = dialog.slice(dialog.indexOf('id="gitCommitModal"'));
        expect(popup).toContain('<h3 id="gitCommitTitle">Save changes</h3>');
        const note = popup.indexOf('All your changes in this folder will be saved.');
        expect(note).toBeGreaterThan(popup.indexOf('gitCommitTitle'));
        expect(note).toBeLessThan(popup.indexOf('<label'));
        expect(popup).toContain('>Message</label>');
        expect(popup).toContain('placeholder="Describe your changes (required)"');
        expect(popup).toContain('id="gitCommitConfirm" disabled>Save</button>');
    });

    it('Save stays off until a message is typed', () => {
        const js = fs.readFileSync(require.resolve('../collectra/gui/sourceControl.js'), 'utf8');
        expect(js).toContain("confirm.disabled = busy || !message.value.trim();");
    });

    it('its error line is hidden until there is an error, so no empty red bar shows', () => {
        expect(dialog).toMatch(/<div id="gitCommitError" class="modal-field-error" role="alert" hidden><\/div>/);
    });

    it('uses no git words a user would not know', () => {
        const words = text(dialog).toLowerCase();
        for (const word of ['commit', 'push', 'pull', 'staged', 'repository', 'source control', 'branch'])
            expect(words, word).not.toContain(word);
    });

    it('the toolbar button says History and Changes too', () => {
        expect(html).toContain('id="gitToolbarBtn" title="History and Changes"');
    });
});

describe('History and Changes dialog fonts', () => {
    const rule = (selector) => {
        const start = html.indexOf(`        ${selector} {`);
        expect(start, selector).toBeGreaterThan(-1);
        return html.slice(start, html.indexOf('}', start));
    };

    it('the result text uses the app font at the same size as the buttons, not a typewriter font', () => {
        const output = rule('.git-output');
        expect(output).toContain('font: inherit;');
        expect(output).toContain('font-size: 13px;');
        expect(output).not.toContain('monospace');
    });

    it('the folder line and the message box match the buttons too', () => {
        expect(rule('.git-repository')).toContain('font-size: 13px;');
        const message = rule('.git-message');
        expect(message).toContain('font: inherit;');
        expect(message).toContain('font-size: 13px;');
    });
});

describe('History and Changes dialog buttons', () => {
    const start = html.indexOf('        .git-actions button, #gitCommitModal .modal-buttons button {');
    const rule = html.slice(start, html.indexOf('}', start));
    const item = html.slice(html.indexOf('        .sidebar-item {'), html.indexOf('}', html.indexOf('        .sidebar-item {')));

    it('have the same corners and spacing as the page list on the left', () => {
        expect(start).toBeGreaterThan(-1);
        const take = (css, name) => css.match(new RegExp(`${name}:\\s*([^;]+);`))[1];
        expect(take(rule, 'border-radius')).toBe(take(item, 'border-radius'));
        expect(take(rule, 'padding')).toBe(take(item, 'padding'));
    });

    it('leave the buttons in other dialogs as they were', () => {
        const shared = html.slice(html.indexOf('        .modal-buttons button {'), html.indexOf('}', html.indexOf('        .modal-buttons button {')));
        expect(shared).toContain('border-radius: 2px;');
        expect(shared).toContain('padding: 5px 14px;');
    });
});

describe('History and Changes close button', () => {
    const block = (selector) => {
        const start = html.indexOf(`        ${selector} {`);
        expect(start, selector).toBeGreaterThan(-1);
        return html.slice(start, html.indexOf('}', start));
    };
    const take = (css, name) => css.match(new RegExp(`${name}:\\s*([^;]+);`))[1];

    it('looks like the toolbar icons: same size, corners and colour, no outline box', () => {
        const close = block('#gitClose');
        const icon = block('.panel-toolbar-btn');
        for (const name of ['width', 'height', 'border-radius', 'color', 'background', 'border'])
            expect(take(close, name), name).toBe(take(icon, name));
    });

    it('lights up on hover like the toolbar icons', () => {
        const hover = block('#gitClose:hover');
        expect(hover).toContain('background: rgba(var(--app-secondary-rgb, 108, 117, 125), 0.2);');
    });

    it('leaves the close buttons of the other dialogs alone', () => {
        const shared = block('.modal-close');
        expect(shared).toContain('font-size: 18px;');
        expect(shared).toContain('background: none;');
    });
});

