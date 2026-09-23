// Per-panel RGB-view navigation. Loading and rendering are injected so the
// boundary, failure, and stale-response behavior can be tested without a DOM.
function createRgbViewNavigation({view, loadView, applyView, isCurrent, onState}) {
    let index = view.rgb_view_index || 0;
    const count = view.rgb_view_count || 1;
    let busy = false;
    let error = '';
    const state = () => ({index, count, busy, error,
        previousDisabled: busy || index === 0,
        nextDisabled: busy || index === count - 1});
    const emit = () => onState(state());
    async function goTo(next) {
        if (busy || !Number.isInteger(next) || next === index || next < 0 || next >= count || !isCurrent()) return;
        busy = true;
        error = '';
        emit();
        try {
            const result = await loadView(next);
            if (!isCurrent()) return;
            if (!result.success) throw new Error(result.error || 'Could not load RGB view');
            await applyView(result.view);
            if (!isCurrent()) return;
            index = next;
        } catch (failure) {
            if (isCurrent()) error = failure.message || String(failure);
        } finally {
            busy = false;
            if (isCurrent()) emit();
        }
    }
    emit();
    return {previous: () => goTo(index - 1), next: () => goTo(index + 1), goTo, state};
}

function mountRgbViewControls({viewer, view, loadView, applyView, isCurrent}) {
    if ((view.rgb_view_count || 1) <= 1) return;
    const bar = document.createElement('div');
    bar.className = 'rgb-view-controls';
    bar.setAttribute('role', 'group');
    bar.setAttribute('aria-label', 'RGB views');
    const previous = document.createElement('button');
    previous.type = 'button';
    previous.textContent = '‹';
    previous.setAttribute('aria-label', 'Previous RGB view');
    previous.title = 'Previous RGB view';
    const counter = document.createElement('select');
    counter.setAttribute('aria-label', 'Choose RGB view');
    const count = view.rgb_view_count;
    // Reserve enough space for the widest label, including the native arrow.
    counter.style.width = `${12 + 2 * String(count).length}ch`;
    for (let index = 0; index < count; index++) {
        const option = document.createElement('option');
        option.value = String(index);
        option.textContent = `View ${index + 1} of ${count}`;
        counter.append(option);
    }
    const next = document.createElement('button');
    next.type = 'button';
    next.textContent = '›';
    next.setAttribute('aria-label', 'Next RGB view');
    next.title = 'Next RGB view';
    const message = document.createElement('span');
    message.className = 'rgb-view-status';
    message.setAttribute('role', 'status');
    bar.append(previous, counter, next, message);
    const navigation = createRgbViewNavigation({view, loadView, applyView, isCurrent,
        onState: state => {
            previous.disabled = state.previousDisabled;
            next.disabled = state.nextDisabled;
            counter.value = String(state.index);
            counter.disabled = state.busy;
            bar.setAttribute('aria-busy', String(state.busy));
            message.textContent = state.error || (state.busy ? 'Loading…' : '');
        },
    });
    previous.onclick = () => navigation.previous();
    next.onclick = () => navigation.next();
    counter.onchange = () => navigation.goTo(Number(counter.value));
    for (const event of ['pointerdown', 'mousedown', 'touchstart', 'click', 'keydown']) {
        bar.addEventListener(event, e => e.stopPropagation());
    }
    viewer.addControl(bar, {anchor: OpenSeadragon.ControlAnchor.TOP_RIGHT});
    return navigation;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {createRgbViewNavigation, mountRgbViewControls};
}
