const {mountShiftDraw} = require('../collectra/gui/shiftDraw.js');
function events() {
    const handlers = new Map();
    return {addEventListener: (key, fn) => handlers.set(key, fn), removeEventListener: key => handlers.delete(key),
        emit: (key, event = {}) => handlers.get(key)?.(event), handlers};
}
function setup(previous = false, allowed = true) {
    const element = events(), document = events(), window = events();
    let drawing = previous;
    const setDrawing = vi.fn(value => {drawing = value;});
    const cleanup = mountShiftDraw({element, document, window, canDraw: () => allowed,
        isDrawing: () => drawing, setDrawing, sync: vi.fn()});
    const down = {button: 0, shiftKey: true, pointerId: 1, pointerType: 'mouse', target: {closest: () => null}};
    return {element, document, window, setDrawing, cleanup, down};
}

it('arms before dragging and restores panning after the drawing event finishes', () => {
    vi.useFakeTimers();
    const s = setup();
    s.element.emit('pointerdown', s.down);
    expect(s.setDrawing).toHaveBeenLastCalledWith(true);
    s.document.emit('pointerup', {pointerId: 1});
    expect(s.setDrawing).toHaveBeenCalledTimes(1);
    vi.runAllTimers();
    expect(s.setDrawing).toHaveBeenLastCalledWith(false);
    s.cleanup();
    vi.useRealTimers();
});

it('preserves a tool already armed through the toolbar', () => {
    vi.useFakeTimers();
    const s = setup(true);
    s.element.emit('pointerdown', s.down);
    s.document.emit('pointerup', {pointerId: 1});
    vi.runAllTimers();
    expect(s.setDrawing).toHaveBeenLastCalledWith(true);
    s.cleanup();
    vi.useRealTimers();
});

it('ignores normal drags, unavailable drawing, and toolbar controls', () => {
    const s = setup();
    s.element.emit('pointerdown', {...s.down, shiftKey: false});
    s.element.emit('pointerdown', {...s.down, button: 2});
    s.element.emit('pointerdown', {...s.down, target: {closest: () => ({})}});
    expect(s.setDrawing).not.toHaveBeenCalled();
    const disabled = setup(false, false);
    disabled.element.emit('pointerdown', disabled.down);
    expect(disabled.setDrawing).not.toHaveBeenCalled();
});

it('cancels on Escape and cleans up event handlers', () => {
    const s = setup();
    s.element.emit('pointerdown', s.down);
    s.document.emit('keydown', {key: 'Escape'});
    expect(s.setDrawing).toHaveBeenLastCalledWith(false);
    s.cleanup();
    expect(s.element.handlers.size).toBe(0);
    expect(s.document.handlers.size).toBe(0);
    expect(s.window.handlers.size).toBe(0);
});

it('prepares the drawing overlay when Shift is held before the gesture', () => {
    const s = setup();
    s.element.emit('pointerenter', {});
    s.document.emit('keydown', {key: 'Shift'});
    expect(s.setDrawing).toHaveBeenLastCalledWith(true);
    s.document.emit('keyup', {key: 'Shift'});
    expect(s.setDrawing).toHaveBeenLastCalledWith(false);
    s.element.emit('pointerenter', {shiftKey: true});
    expect(s.setDrawing).toHaveBeenLastCalledWith(true);
    s.element.emit('pointerleave');
    expect(s.setDrawing).toHaveBeenLastCalledWith(false);
});
