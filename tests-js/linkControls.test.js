const {linkMenuItems, createLinkEditor, mountLinkControls} = require('../collectra/gui/linkControls.js');

it('lists the names to choose from, the current one marked, then Undefined', () => {
    expect(linkMenuItems('handwritten', ['printed', 'handwritten', 'combination'])).toEqual([
        {name: 'combination', current: false},
        {name: 'handwritten', current: true},
        {name: 'printed', current: false},
        {undefined: true},
    ]);
});

it('offers the current name even when no list has it', () => {
    expect(linkMenuItems('odd', ['printed'])).toEqual([
        {name: 'odd', current: true},
        {name: 'printed', current: false},
        {undefined: true},
    ]);
});

it('keeps a task\'s classes in their own order and has no Undefined when nothing is set', () => {
    expect(linkMenuItems(null, ['printed', 'handwritten'], {sorted: false})).toEqual([
        {name: 'printed', current: false},
        {name: 'handwritten', current: false},
    ]);
});

function editor(overrides = {}) {
    const options = {
        link: {id: 'handwritten', name: 'handwritten'},
        loadNames: vi.fn(async () => ({success: true, names: ['handwritten', 'printed']})),
        rename: vi.fn(async (id, name) => ({success: true, id: name})),
        create: vi.fn(async (name) => ({success: true, id: name})),
        remove: vi.fn(async () => true),
        onChanged: vi.fn(async () => {}),
        isCurrent: () => true,
        onState: vi.fn(),
        ...overrides,
    };
    return {editor: createLinkEditor(options), ...options};
}

it('starts closed, showing the link name', () => {
    const {editor: e} = editor();
    expect(e.state()).toMatchObject({id: 'handwritten', name: 'handwritten', open: false, busy: false, error: ''});
});

it('loads the names when opened and closes again', async () => {
    const {editor: e, loadNames} = editor();
    await e.toggle();
    expect(loadNames).toHaveBeenCalledTimes(1);
    expect(e.state()).toMatchObject({open: true, names: ['handwritten', 'printed']});
    await e.toggle();
    expect(e.state().open).toBe(false);
});

it('still opens, with the current name, when the names cannot be loaded', async () => {
    const {editor: e} = editor({loadNames: vi.fn(async () => ({success: false, error: 'no page'}))});
    await e.toggle();
    expect(e.state()).toMatchObject({open: true, names: ['handwritten'], error: 'no page'});
});

it('renames the link, takes the id it is given back and reports the change', async () => {
    const {editor: e, rename, onChanged} = editor();
    await e.toggle();
    await e.choose('printed');
    expect(rename).toHaveBeenCalledWith('handwritten', 'printed');
    expect(e.state()).toMatchObject({id: 'printed', name: 'printed', open: false, busy: false, error: ''});
    expect(onChanged).toHaveBeenCalledTimes(1);
});

it('does nothing when the current name is chosen', async () => {
    const {editor: e, rename} = editor();
    await e.toggle();
    await e.choose('handwritten');
    expect(rename).not.toHaveBeenCalled();
    expect(e.state().open).toBe(false);
});

it('ignores a name that is only spaces', async () => {
    const {editor: e, rename} = editor();
    await e.choose('   ');
    expect(rename).not.toHaveBeenCalled();
});

it('keeps the old name and says why when the rename fails', async () => {
    const {editor: e, onChanged} = editor({rename: vi.fn(async () => ({success: false, error: 'boom'}))});
    await e.choose('printed');
    expect(e.state()).toMatchObject({name: 'handwritten', error: 'boom', busy: false});
    expect(onChanged).not.toHaveBeenCalled();
});

it('keeps the old name when the rename throws', async () => {
    const {editor: e} = editor({rename: vi.fn(async () => { throw new Error('offline'); })});
    await e.choose('printed');
    expect(e.state()).toMatchObject({name: 'handwritten', error: 'offline', busy: false});
});

it('works on one change at a time', async () => {
    let finish;
    const {editor: e, rename} = editor({rename: vi.fn(() => new Promise(resolve => {finish = resolve;}))});
    const first = e.choose('printed');
    expect(e.state().busy).toBe(true);
    await e.choose('other');
    expect(rename).toHaveBeenCalledTimes(1);
    finish({success: true, id: 'printed'});
    await first;
    expect(e.state().busy).toBe(false);
});

it('removes the link when set to Undefined, after the remove step confirms it', async () => {
    const {editor: e, remove, onChanged} = editor();
    await e.toggle();
    await e.undefine();
    expect(remove).toHaveBeenCalledWith('handwritten', 'handwritten');
    expect(onChanged).toHaveBeenCalledTimes(1);
    expect(e.state()).toMatchObject({open: false, busy: false});
});

it('leaves the link alone when the removal is declined', async () => {
    const {editor: e, onChanged} = editor({remove: vi.fn(async () => false)});
    await e.undefine();
    expect(onChanged).not.toHaveBeenCalled();
    expect(e.state()).toMatchObject({name: 'handwritten', busy: false, error: ''});
});

it('reports a removal that throws', async () => {
    const {editor: e} = editor({remove: vi.fn(async () => { throw new Error('nope'); })});
    await e.undefine();
    expect(e.state()).toMatchObject({error: 'nope', busy: false});
});

it('ignores a response that arrives after leaving the artefact', async () => {
    let current = true;
    let finish;
    const {editor: e, onChanged, onState} = editor({isCurrent: () => current,
        rename: vi.fn(() => new Promise(resolve => {finish = resolve;}))});
    const pending = e.choose('printed');
    current = false;
    onState.mockClear();
    finish({success: true, id: 'printed'});
    await pending;
    expect(onChanged).not.toHaveBeenCalled();
    expect(onState).not.toHaveBeenCalled();
});

const slot = {id: null, name: null, task: 'classify', options: ['printed', 'handwritten']};

it('offers a task\'s classes without loading names, even when nothing is set', async () => {
    const {editor: e, loadNames} = editor({link: slot});
    expect(e.state()).toMatchObject({id: null, name: null, task: 'classify', closed: true});
    await e.toggle();
    expect(loadNames).not.toHaveBeenCalled();
    expect(e.state()).toMatchObject({open: true, names: ['printed', 'handwritten']});
});

it('creates the link when a class is chosen and none was set', async () => {
    const {editor: e, create, rename, onChanged} = editor({link: slot});
    await e.toggle();
    await e.choose('handwritten');
    expect(create).toHaveBeenCalledWith('handwritten');
    expect(rename).not.toHaveBeenCalled();
    expect(e.state()).toMatchObject({id: 'handwritten', name: 'handwritten', open: false, error: ''});
    expect(onChanged).toHaveBeenCalledTimes(1);
});

it('renames the link on a later choice, once it exists', async () => {
    const {editor: e, rename} = editor({link: slot});
    await e.choose('handwritten');
    await e.choose('printed');
    expect(rename).toHaveBeenCalledWith('handwritten', 'printed');
});

it('says why when the link cannot be created', async () => {
    const {editor: e, onChanged} = editor({link: slot, create: vi.fn(async () => ({success: false, error: 'boom'}))});
    await e.choose('printed');
    expect(e.state()).toMatchObject({id: null, name: null, error: 'boom'});
    expect(onChanged).not.toHaveBeenCalled();
});

it('has nothing to remove when no link is set', async () => {
    const {editor: e, remove} = editor({link: slot});
    await e.undefine();
    expect(remove).not.toHaveBeenCalled();
});

it('ignores a link created after leaving the artefact', async () => {
    let current = true;
    let finish;
    const {editor: e, onChanged} = editor({link: slot, isCurrent: () => current,
        create: vi.fn(() => new Promise(resolve => {finish = resolve;}))});
    const pending = e.choose('printed');
    current = false;
    finish({success: true, id: 'printed'});
    await pending;
    expect(onChanged).not.toHaveBeenCalled();
});

function fakeDom() {
    const element = tag => ({
        tag, children: [], attributes: {}, style: {}, dataset: {}, hidden: false, textContent: '',
        classList: {add() {}, remove() {}, toggle() {}},
        setAttribute(k, v) {this.attributes[k] = v;},
        append(...kids) {this.children.push(...kids);},
        replaceChildren(...kids) {this.children = kids;},
        addEventListener: vi.fn(),
    });
    vi.stubGlobal('document', {createElement: element, addEventListener: vi.fn(), removeEventListener: vi.fn()});
    vi.stubGlobal('OpenSeadragon', {ControlAnchor: {TOP_LEFT: 1}});
}

function mount(links, overrides = {}) {
    const viewer = {addControl: vi.fn()};
    const options = {
        viewer, view: {links},
        isCurrent: () => true,
        loadNames: async () => ({success: true, names: ['handwritten', 'printed']}),
        rename: vi.fn(async (id, name) => ({success: true, id: name})),
        create: vi.fn(async (name) => ({success: true, id: name})),
        remove: vi.fn(async () => true),
        onChanged: vi.fn(async () => {}),
        ...overrides,
    };
    return {editors: mountLinkControls(options), viewer, ...options};
}

it('adds nothing to an image that has no Link children', () => {
    fakeDom();
    try {
        const {viewer, editors} = mount([]);
        expect(viewer.addControl).not.toHaveBeenCalled();
        expect(editors).toEqual([]);
    } finally {
        vi.unstubAllGlobals();
    }
});

it('adds a button naming each Link, at the top left beside the other tools', async () => {
    fakeDom();
    try {
        const {viewer, editors} = mount([{id: 'a', name: 'handwritten'}, {id: 'b', name: 'printed'}]);
        expect(viewer.addControl).toHaveBeenCalledTimes(1);
        const [bar, placement] = viewer.addControl.mock.calls[0];
        expect(placement.anchor).toBe(1);
        expect(editors).toHaveLength(2);
        const [first, second] = bar.children;
        expect(first.children[0].textContent).toContain('handwritten');
        expect(second.children[0].textContent).toContain('printed');
        await first.children[0].onclick();
        const menu = first.children[1];
        expect(menu.hidden).toBe(false);
        expect(menu.children.map(item => item.textContent)).toEqual(
            ['handwritten', 'printed', 'Undefined']);
    } finally {
        vi.unstubAllGlobals();
    }
});

it('renames through the menu and shows the new name on the button', async () => {
    fakeDom();
    try {
        const {viewer, rename, onChanged} = mount([{id: 'a', name: 'handwritten'}]);
        const [bar] = viewer.addControl.mock.calls[0];
        const [wrapper] = bar.children;
        await wrapper.children[0].onclick();
        await wrapper.children[1].children[1].onclick();
        expect(rename).toHaveBeenCalledWith('a', 'printed');
        expect(wrapper.children[0].textContent).toContain('printed');
        expect(onChanged).toHaveBeenCalled();
    } finally {
        vi.unstubAllGlobals();
    }
});

it('removes through the Undefined item', async () => {
    fakeDom();
    try {
        const {viewer, remove} = mount([{id: 'a', name: 'handwritten'}]);
        const [bar] = viewer.addControl.mock.calls[0];
        const [wrapper] = bar.children;
        await wrapper.children[0].onclick();
        await wrapper.children[1].children[2].onclick();
        expect(remove).toHaveBeenCalledWith('a', 'handwritten');
    } finally {
        vi.unstubAllGlobals();
    }
});

it('shows a task with nothing set by the task name, and creates the link from its classes', async () => {
    fakeDom();
    try {
        const loadNames = vi.fn();
        const {viewer, create, onChanged} = mount([
            {id: null, name: null, task: 'classify', options: ['printed', 'handwritten']},
        ], {loadNames});
        const [bar] = viewer.addControl.mock.calls[0];
        const [wrapper] = bar.children;
        const [button, menu, entry] = wrapper.children;
        expect(button.textContent).toBe('classify: none ▾');
        await button.onclick();
        expect(menu.hidden).toBe(false);
        expect(entry.hidden).toBe(true);
        expect(menu.children.map(item => item.textContent)).toEqual(['printed', 'handwritten']);
        await menu.children[1].onclick();
        expect(loadNames).not.toHaveBeenCalled();
        expect(create).toHaveBeenCalledWith('handwritten');
        expect(button.textContent).toBe('classify: handwritten ▾');
        expect(onChanged).toHaveBeenCalled();
    } finally {
        vi.unstubAllGlobals();
    }
});

it('offers Undefined for a task whose link is set', async () => {
    fakeDom();
    try {
        const {viewer} = mount([
            {id: 'a', name: 'printed', task: 'classify', options: ['printed', 'handwritten']},
        ]);
        const [bar] = viewer.addControl.mock.calls[0];
        const [button, menu] = bar.children[0].children;
        await button.onclick();
        expect(menu.children.map(item => item.textContent)).toEqual(['printed', 'handwritten', 'Undefined']);
    } finally {
        vi.unstubAllGlobals();
    }
});
