const {createRgbViewNavigation, mountRgbViewControls} = require('../collectra/gui/rgbViewControls.js');

function navigation(overrides = {}) {
    const options = {
        view: {rgb_view_count: 25, rgb_view_index: 0},
        loadView: vi.fn(async index => ({success: true, view: {rgb_view_index: index}})),
        applyView: vi.fn(async () => {}),
        isCurrent: () => true,
        onState: vi.fn(),
        ...overrides,
    };
    return {nav: createRgbViewNavigation(options), ...options};
}

it('moves both ways through 25 views and stops at the ends', async () => {
    const {nav, loadView, applyView} = navigation();
    expect(nav.state().previousDisabled).toBe(true);
    await nav.previous();
    expect(loadView).not.toHaveBeenCalled();
    for (let i = 1; i < 25; i++) await nav.next();
    expect(nav.state().index).toBe(24);
    expect(nav.state().nextDisabled).toBe(true);
    await nav.next();
    expect(loadView).toHaveBeenCalledTimes(24);
    await nav.previous();
    expect(applyView).toHaveBeenLastCalledWith({rgb_view_index: 23});
    expect(nav.state().index).toBe(23);
});

it('disables navigation during loading and ignores repeated clicks', async () => {
    let finish;
    const {nav, loadView} = navigation({loadView: vi.fn(() => new Promise(resolve => {finish = resolve;}))});
    const pending = nav.next();
    expect(nav.state().previousDisabled).toBe(true);
    expect(nav.state().nextDisabled).toBe(true);
    await nav.next();
    expect(loadView).toHaveBeenCalledTimes(1);
    finish({success: true, view: {rgb_view_index: 1}});
    await pending;
    expect(nav.state().busy).toBe(false);
});

it('keeps the previous view after a load failure and allows retry', async () => {
    const {nav, applyView, loadView} = navigation({loadView: vi.fn(async () => ({success: false, error: 'Missing file'}))});
    await nav.next();
    expect(nav.state().index).toBe(0);
    expect(nav.state().error).toBe('Missing file');
    expect(applyView).not.toHaveBeenCalled();
    loadView.mockResolvedValue({success: true, view: {rgb_view_index: 1}});
    await nav.next();
    expect(nav.state().index).toBe(1);
    expect(nav.state().error).toBe('');
});

it('does not replace pixels when a response arrives after leaving the artefact', async () => {
    let current = true;
    let finish;
    const {nav, applyView} = navigation({isCurrent: () => current,
        loadView: () => new Promise(resolve => {finish = resolve;})});
    const pending = nav.next();
    current = false;
    finish({success: true, view: {rgb_view_index: 1}});
    await pending;
    expect(applyView).not.toHaveBeenCalled();
});

it('shows buttons and a counter at the top of OpenSeadragon', async () => {
    const element = () => ({children: [], attributes: {}, style: {},
        setAttribute(k, v) {this.attributes[k] = v;},
        append(...children) {this.children.push(...children);},
        addEventListener: vi.fn(),
    });
    vi.stubGlobal('document', {createElement: element});
    vi.stubGlobal('OpenSeadragon', {ControlAnchor: {TOP_RIGHT: 2}});
    try {
        const viewer = {addControl: vi.fn()};
        const nav = mountRgbViewControls({viewer, view: {rgb_view_count: 25},
            isCurrent: () => true, loadView: async index => ({success: true, view: {rgb_view_index: index}}),
            applyView: async () => {}});
        const [bar, placement] = viewer.addControl.mock.calls[0];
        expect(placement.anchor).toBe(2);
        expect(bar.children[0].textContent).toBe('‹');
        expect(bar.children[2].textContent).toBe('›');
        expect(bar.children[1].children).toHaveLength(25);
        expect(bar.children[1].children[24].textContent).toBe('View 25 of 25');
        expect(bar.children[1].value).toBe('0');
        expect(bar.children[0].disabled).toBe(true);
        await bar.children[2].onclick();
        expect(bar.children[1].value).toBe('1');
        expect(nav.state().index).toBe(1);
        bar.children[1].value = '19';
        await bar.children[1].onchange();
        expect(nav.state().index).toBe(19);
        expect(bar.children[1].value).toBe('19');
        viewer.addControl.mockClear();
        mountRgbViewControls({viewer, view: {rgb_view_count: 1}});
        expect(viewer.addControl).not.toHaveBeenCalled();
    } finally {
        vi.unstubAllGlobals();
    }
});


it('jumps directly to a chosen view and ignores invalid or unchanged choices', async () => {
    const {nav, loadView, applyView} = navigation();
    await nav.goTo(24);
    expect(loadView).toHaveBeenCalledTimes(1);
    expect(loadView).toHaveBeenCalledWith(24);
    expect(applyView).toHaveBeenCalledWith({rgb_view_index: 24});
    expect(nav.state().nextDisabled).toBe(true);
    for (const index of [24, -1, 25, 1.5, NaN]) await nav.goTo(index);
    expect(loadView).toHaveBeenCalledTimes(1);
    await nav.goTo(0);
    expect(nav.state().previousDisabled).toBe(true);
});
