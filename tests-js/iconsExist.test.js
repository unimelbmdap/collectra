// Every Font Awesome icon the page uses must exist in the bundled copy of the font
// (version 6). A name from an older version draws as a blank square.
const fs = require('node:fs');
const path = require('node:path');
const gui = path.resolve(__dirname, '../collectra/gui');
const css = fs.readFileSync(path.join(gui, 'vendor/fontawesome/css/all.min.css'), 'utf8');
const sources = ['index.html', ...fs.readdirSync(gui).filter((f) => f.endsWith('.js'))]
    .map((f) => [f, fs.readFileSync(path.join(gui, f), 'utf8')]);

// Style classes and sizing helpers, not icons.
const NOT_ICONS = new Set(['fa', 'fa-solid', 'fa-regular', 'fa-brands', 'fa-fw', 'fa-lg', 'fa-spin', 'fa-2x']);
const defined = (name) => new RegExp(`\\.${name}(:before|,|\\{)`).test(css);

describe('Font Awesome icons', () => {
    for (const [file, text] of sources) {
        const names = [...new Set(text.match(/\bfa-[a-z0-9-]+/g) || [])].filter((n) => !NOT_ICONS.has(n));
        if (!names.length) continue;
        it(`${file} only uses icons the bundled font has`, () => {
            const missing = names.filter((n) => !defined(n));
            expect(missing).toEqual([]);
        });
    }

    it('the delete button is a bin', () => {
        const html = sources.find(([f]) => f === 'index.html')[1];
        const at = html.indexOf('const NODE_ACTION_ICONS');
        const icons = html.slice(at, html.indexOf('\n        };', at));
        expect(icons).toContain('fa-trash-can');
    });
});
