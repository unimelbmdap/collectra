const fs = require('node:fs');
const {envRows, envNameProblem, setupSettings} = require('../collectra/gui/settingsDialog.js');

const html = fs.readFileSync(require.resolve('../collectra/gui/index.html'), 'utf8');
const dialog = html.slice(html.indexOf('id="settingsModal"'), html.indexOf('<!-- New-box Popup Modal -->'));

const variable = (name, extra = {}) => ({
    name, value: 'v', private: false, overridden: false, override: false, replaces_shell: false, missing: false, ...extra,
});
const listing = (variables, extra = {}) => ({
    success: true, pipeline: 'hespi', path: '/cfg/env/hespi.yaml', keyring_available: true, variables, ...extra,
});

describe('envRows (one row per stored variable)', () => {
    it('shows each value and says where it is kept and whether it is used', () => {
        expect(envRows(listing([
            variable('HF_HOME', {value: '/data/hf'}),
            variable('API_KEY', {value: 'sk-a…efgh', private: true}),
            variable('TOKEN', {overridden: true}),
            variable('PATHS', {override: true, replaces_shell: true}),
            variable('GONE', {value: '', private: true, missing: true}),
        ]))).toEqual([
            {name: 'HF_HOME', value: '/data/hf', storage: 'File', status: 'In use', state: 'used', override: false},
            {name: 'API_KEY', value: 'sk-a…efgh', storage: 'Keyring', status: 'In use', state: 'used', override: false},
            {name: 'TOKEN', value: 'v', storage: 'File', status: 'Overridden by the shell', state: 'overridden', override: false},
            {name: 'PATHS', value: 'v', storage: 'File', status: 'Overrides the shell', state: 'replacing', override: true},
            {name: 'GONE', value: '', storage: 'Keyring', status: 'Not in the keyring on this computer', state: 'missing', override: false},
        ]);
    });

    it('has no rows when the listing has none or failed', () => {
        for (const empty of [null, undefined, {}, listing([]), {success: false, variables: []}]) {
            expect(envRows(empty)).toEqual([]);
        }
    });
});

describe('envNameProblem (checking a name before it is sent)', () => {
    it('accepts environment variable names', () => {
        for (const name of ['ANTHROPIC_API_KEY', '_X', 'a1']) expect(envNameProblem(name)).toBe('');
    });

    it('explains what is wrong with any other name', () => {
        expect(envNameProblem('')).toBe('');  // nothing typed yet is not an error
        for (const name of ['1ABC', 'MY-KEY', 'A B']) {
            expect(envNameProblem(name)).toBe('Use letters, digits and _, not starting with a digit.');
        }
    });
});

// A stand-in for the few parts of the DOM the dialog uses.
class Element {
    constructor(tag, id = '') {
        this.tagName = tag.toUpperCase();
        this.id = id;
        this.children = [];
        this.textContent = '';
        this.hidden = false;
        this.disabled = false;
        this.value = '';
        this.checked = false;
        this.type = '';
        this.className = '';
        this.dataset = {};
        this.attributes = {};
        this.listeners = {};
        const classes = new Set();
        this.classList = {
            add: (name) => classes.add(name),
            remove: (name) => classes.delete(name),
            contains: (name) => classes.has(name),
        };
    }
    append(...children) { this.children.push(...children); }
    replaceChildren(...children) { this.children = children; }
    setAttribute(name, value) { this.attributes[name] = String(value); }
    addEventListener(type, listener) { (this.listeners[type] ||= []).push(listener); }
    focus() { this.document.activeElement = this; }
    fire(type, extra = {}) {
        const event = {type, key: extra.key, preventDefault: () => { event.prevented = true; }};
        for (const listener of this.listeners[type] || []) listener(event);
        return event;
    }
    all() { return [this, ...this.children.flatMap((child) => child.all())]; }
    text() { return [this.textContent, ...this.children.map((child) => child.text())].join(' '); }
}

const IDS = [
    'settingsToolbarBtn', 'settingsModal', 'settingsClose', 'envTable', 'envRows', 'envEmpty',
    'envPath', 'envAddForm', 'envNameInput', 'envValueInput', 'envPrivateInput', 'envPrivateNote',
    'envOverrideInput',
    'envAddBtn', 'envError', 'envClearBtn', 'envClearConfirm', 'envClearQuestion', 'envClearYes',
    'envClearNo',
];

function page(responses) {
    const document = {listeners: {}, activeElement: null};
    const elements = Object.fromEntries(IDS.map((id) => [id, new Element('div', id)]));
    for (const element of Object.values(elements)) element.document = document;
    document.getElementById = (id) => elements[id] || null;
    document.createElement = (tag) => Object.assign(new Element(tag), {document});
    document.addEventListener = (type, listener) => { (document.listeners[type] ||= []).push(listener); };
    document.key = (key) => {
        const event = {key, preventDefault: () => { event.prevented = true; }};
        for (const listener of document.listeners.keydown || []) listener(event);
        return event;
    };
    const calls = [];
    const api = new Proxy({}, {
        get: (_, method) => async (...args) => {
            calls.push([method, ...args]);
            return typeof responses[method] === 'function' ? responses[method](...args) : responses[method];
        },
    });
    setupSettings(document, () => api);
    return {document, elements, calls};
}

const settle = () => new Promise((resolve) => setTimeout(resolve));

async function open(responses) {
    const view = page(responses);
    view.elements.settingsToolbarBtn.fire('click');
    await settle();
    return view;
}

const rowNames = ({elements}) => elements.envRows.children.map((row) => row.children[0].textContent);
const deleteButton = ({elements}, name) => elements.envRows.all()
    .find((element) => element.tagName === 'BUTTON' && element.dataset.name === name);
const overrideBox = ({elements}, name) => elements.envRows.all()
    .find((element) => element.tagName === 'INPUT' && element.dataset.name === name);

describe('the Settings dialog', () => {
    it('opens from the cog and lists the stored variables', async () => {
        const view = await open({get_env_variables: listing([variable('HF_HOME', {value: '/data/hf'})])});
        expect(view.elements.settingsModal.classList.contains('visible')).toBe(true);
        expect(view.calls).toEqual([['get_env_variables']]);
        expect(rowNames(view)).toEqual(['HF_HOME']);
        expect(view.elements.envRows.text()).toContain('/data/hf');
        expect(view.elements.envTable.hidden).toBe(false);
        expect(view.elements.envEmpty.hidden).toBe(true);
        expect(view.elements.envPath.textContent).toBe('Stored for hespi in /cfg/env/hespi.yaml');
        expect(view.elements.envClearBtn.disabled).toBe(false);
    });

    it('marks overridden variables so they stand out', async () => {
        const view = await open({get_env_variables: listing([variable('TOKEN', {overridden: true})])});
        const [row] = view.elements.envRows.children;
        expect(row.className).toBe('env-row env-overridden');
        expect(row.text()).toContain('Overridden by the shell');
    });

    it('says when nothing is stored, and Clear all is off', async () => {
        const view = await open({get_env_variables: listing([])});
        expect(view.elements.envTable.hidden).toBe(true);
        expect(view.elements.envEmpty.hidden).toBe(false);
        expect(view.elements.envClearBtn.disabled).toBe(true);
    });

    it('shows an error from the backend', async () => {
        const view = await open({get_env_variables: {success: false, error: 'Cannot read hespi.yaml', variables: []}});
        expect(view.elements.envError.hidden).toBe(false);
        expect(view.elements.envError.textContent).toBe('Cannot read hespi.yaml');
    });

    it('adds a variable, then clears the form', async () => {
        const view = await open({
            get_env_variables: listing([]),
            set_env_variable: listing([variable('HF_HOME', {value: '/data/hf'})]),
        });
        const {envNameInput, envValueInput, envAddBtn, envAddForm} = view.elements;
        envNameInput.value = 'HF_HOME';
        envNameInput.fire('input');
        expect(envAddBtn.disabled).toBe(true);  // no value yet
        envValueInput.value = '/data/hf';
        envValueInput.fire('input');
        expect(envAddBtn.disabled).toBe(false);
        const submitted = envAddForm.fire('submit');
        await settle();
        expect(submitted.prevented).toBe(true);
        expect(view.calls.at(-1)).toEqual(['set_env_variable', 'HF_HOME', '/data/hf', false, false]);
        expect(rowNames(view)).toEqual(['HF_HOME']);
        expect(envNameInput.value).toBe('');
        expect(envValueInput.value).toBe('');
        expect(envAddBtn.disabled).toBe(true);
        expect(view.document.activeElement).toBe(envNameInput);
    });

    it('keeps what was typed when the backend refuses it', async () => {
        const view = await open({
            get_env_variables: listing([]),
            set_env_variable: listing([], {success: false, error: 'No secure keyring'}),
        });
        const {envNameInput, envValueInput, envPrivateInput, envAddForm, envError} = view.elements;
        envNameInput.value = 'API_KEY';
        envValueInput.value = 'sk-x';
        envPrivateInput.checked = true;
        view.elements.envOverrideInput.checked = true;
        envAddForm.fire('submit');
        await settle();
        expect(view.calls.at(-1)).toEqual(['set_env_variable', 'API_KEY', 'sk-x', true, true]);
        expect(envError.textContent).toBe('No secure keyring');
        expect(envNameInput.value).toBe('API_KEY');
    });

    it('explains a bad name as it is typed, and will not send it', async () => {
        const view = await open({get_env_variables: listing([])});
        const {envNameInput, envValueInput, envAddBtn, envAddForm, envError} = view.elements;
        envValueInput.value = 'x';
        envNameInput.value = 'MY-KEY';
        envNameInput.fire('input');
        expect(envError.textContent).toBe('Use letters, digits and _, not starting with a digit.');
        expect(envAddBtn.disabled).toBe(true);
        envAddForm.fire('submit');
        await settle();
        expect(view.calls).toEqual([['get_env_variables']]);
        envNameInput.value = 'MY_KEY';
        envNameInput.fire('input');
        expect(envError.hidden).toBe(true);
    });

    it('hides a private value as it is typed', async () => {
        const view = await open({get_env_variables: listing([])});
        const {envPrivateInput, envValueInput} = view.elements;
        expect(envValueInput.type).toBe('text');
        envPrivateInput.checked = true;
        envPrivateInput.fire('change');
        expect(envValueInput.type).toBe('password');
        envPrivateInput.checked = false;
        envPrivateInput.fire('change');
        expect(envValueInput.type).toBe('text');
    });

    it('turns Private off, and says why, without a keyring', async () => {
        const view = await open({get_env_variables: listing([], {keyring_available: false})});
        const {envPrivateInput, envPrivateNote} = view.elements;
        expect(envPrivateInput.disabled).toBe(true);
        expect(envPrivateInput.checked).toBe(false);
        expect(envPrivateNote.hidden).toBe(false);
    });

    it('deletes one variable from its row', async () => {
        const view = await open({
            get_env_variables: listing([variable('A'), variable('B')]),
            delete_env_variable: listing([variable('B')]),
        });
        const button = deleteButton(view, 'A');
        expect(button.attributes['aria-label']).toBe('Delete A');
        button.fire('click');
        await settle();
        expect(view.calls.at(-1)).toEqual(['delete_env_variable', 'A']);
        expect(rowNames(view)).toEqual(['B']);
    });

    it('switches a variable\'s override from its row', async () => {
        const view = await open({
            get_env_variables: listing([variable('A', {overridden: true})]),
            set_env_override: listing([variable('A', {override: true, replaces_shell: true})]),
        });
        const box = overrideBox(view, 'A');
        expect(box.type).toBe('checkbox');
        expect(box.checked).toBe(false);
        expect(box.attributes['aria-label']).toBe('A overrides the shell');
        box.checked = true;
        box.fire('change');
        await settle();
        expect(view.calls.at(-1)).toEqual(['set_env_override', 'A', true]);
        expect(overrideBox(view, 'A').checked).toBe(true);
        expect(view.elements.envRows.text()).toContain('Overrides the shell');
    });

    it('asks before clearing everything', async () => {
        const view = await open({
            get_env_variables: listing([variable('A'), variable('B')]),
            clear_env_variables: listing([]),
        });
        const {envClearBtn, envClearConfirm, envClearQuestion, envClearYes, envClearNo} = view.elements;
        envClearBtn.fire('click');
        expect(envClearConfirm.hidden).toBe(false);
        expect(envClearQuestion.textContent).toBe('Delete all 2 variables?');
        envClearNo.fire('click');
        expect(envClearConfirm.hidden).toBe(true);
        expect(view.calls).toEqual([['get_env_variables']]);

        envClearBtn.fire('click');
        envClearYes.fire('click');
        await settle();
        expect(view.calls.at(-1)).toEqual(['clear_env_variables']);
        expect(envClearConfirm.hidden).toBe(true);
        expect(rowNames(view)).toEqual([]);
    });

    it('says "1 variable" for one', async () => {
        const view = await open({get_env_variables: listing([variable('A')])});
        view.elements.envClearBtn.fire('click');
        expect(view.elements.envClearQuestion.textContent).toBe('Delete 1 variable?');
    });

    it('closes with the close button or Escape and gives focus back', async () => {
        const view = await open({get_env_variables: listing([])});
        const {settingsModal, settingsClose, settingsToolbarBtn} = view.elements;
        expect(view.document.activeElement).toBe(view.elements.envNameInput);
        settingsClose.fire('click');
        expect(settingsModal.classList.contains('visible')).toBe(false);

        view.document.activeElement = settingsToolbarBtn;
        settingsToolbarBtn.fire('click');
        await settle();
        expect(view.document.key('Enter').prevented).toBeUndefined();
        expect(view.document.key('Escape').prevented).toBe(true);
        expect(settingsModal.classList.contains('visible')).toBe(false);
        expect(view.document.activeElement).toBe(settingsToolbarBtn);
        expect(view.document.key('Escape').prevented).toBeUndefined();
    });

    it('Escape first backs out of Clear all', async () => {
        const view = await open({get_env_variables: listing([variable('A')])});
        view.elements.envClearBtn.fire('click');
        view.document.key('Escape');
        expect(view.elements.envClearConfirm.hidden).toBe(true);
        expect(view.elements.settingsModal.classList.contains('visible')).toBe(true);
    });

    it('reports a failed call', async () => {
        const view = await open({get_env_variables: () => { throw new Error('bridge gone'); }});
        expect(view.elements.envError.textContent).toBe('Error: bridge gone');
    });
});

describe('the Settings markup', () => {
    it('has a cog button in the toolbar', () => {
        expect(html).toContain('id="settingsToolbarBtn" title="Settings" aria-label="Settings"');
        expect(html).toContain('<script src="settingsDialog.js"></script>');
    });

    it('has every element the dialog uses, inside the dialog', () => {
        for (const id of IDS.filter((id) => id !== 'settingsToolbarBtn' && id !== 'settingsModal')) {
            expect(dialog, id).toContain(`id="${id}"`);
        }
    });

    it('is titled Settings, with a section for environment variables', () => {
        expect(dialog).toContain('<h3 id="settingsTitle">Settings</h3>');
        expect(dialog).toContain('Environment variables');
        expect(dialog).toContain('aria-label="Close settings"');
    });

    it('hides its error line and the Clear all question until needed', () => {
        expect(dialog).toMatch(/id="envError" class="modal-field-error" role="alert" hidden>/);
        expect(dialog).toMatch(/id="envClearConfirm"[^>]* hidden>/);
    });

    it('can be opened from the Tools menu', () => {
        const menu = fs.readFileSync(require.resolve('../collectra/gui/nativeMenu.js'), 'utf8');
        expect(menu).toContain("settings: () => click('settingsToolbarBtn'),");
    });
});
