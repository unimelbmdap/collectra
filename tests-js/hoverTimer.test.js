const fs = require('node:fs');
const { scheduleHoverHide, cancelHoverHide, HOVER_HIDE_MS } = require('../collectra/gui/hoverTimer.js');
const html = fs.readFileSync(require.resolve('../collectra/gui/index.html'), 'utf8');
const between = (from, to) => html.slice(html.indexOf(from), html.indexOf(to, html.indexOf(from)));

beforeEach(() => vi.useFakeTimers());
afterEach(() => vi.useRealTimers());

describe('hover cards go away by themselves', () => {
    it('after four seconds', () => {
        expect(HOVER_HIDE_MS).toBe(4000);
        const card = { style: { display: 'block' } };
        scheduleHoverHide(card);
        vi.advanceTimersByTime(3999);
        expect(card.style.display).toBe('block');
        vi.advanceTimersByTime(1);
        expect(card.style.display).toBe('none');
    });

    it('showing it again restarts the time instead of cutting the new one short', () => {
        const card = { style: { display: 'block' } };
        scheduleHoverHide(card);
        vi.advanceTimersByTime(3000);
        scheduleHoverHide(card);
        vi.advanceTimersByTime(3000);
        expect(card.style.display).toBe('block');
        vi.advanceTimersByTime(1000);
        expect(card.style.display).toBe('none');
    });

    it('leaving cancels the timer, so it cannot hide a card shown later', () => {
        const card = { style: { display: 'block' } };
        scheduleHoverHide(card);
        cancelHoverHide(card);
        vi.advanceTimersByTime(10000);
        expect(card.style.display).toBe('block');
    });

    it('each card has its own timer', () => {
        const a = { style: { display: 'block' } };
        const b = { style: { display: 'block' } };
        scheduleHoverHide(a);
        vi.advanceTimersByTime(2000);
        scheduleHoverHide(b);
        vi.advanceTimersByTime(2000);
        expect(a.style.display).toBe('none');
        expect(b.style.display).toBe('block');
    });

    it('copes with no card', () => {
        expect(() => { scheduleHoverHide(null); cancelHoverHide(undefined); }).not.toThrow();
    });
});

describe('in the page', () => {
    it('loads the timer', () => {
        expect(html).toContain('<script src="hoverTimer.js"></script>');
    });

    it('the tab hover times out, and is dropped when the tabs are redrawn', () => {
        expect(between('function showTabTooltip(', 'function hideTabTooltip(')).toContain('scheduleHoverHide(tooltip);');
        expect(between('function hideTabTooltip(', '// ── END TAB TOOLTIP')).toContain('cancelHoverHide(');
        expect(between('function renderTabStrip(', 'const hit =')).toContain('hideTabTooltip();');
    });

    it('the image box hover and the DAG hover time out too', () => {
        expect(between("this.pageAnno.on('mouseEnterAnnotation'", "this.pageAnno.on('mouseLeaveAnnotation'")).toContain('scheduleHoverHide(tooltip);');
        expect(between('function showDagTooltip(', 'function hideDagTooltip(')).toContain('scheduleHoverHide(tooltip);');
        expect(between('function hideDagTooltip(', 'function moveDagTooltip(')).toContain('cancelHoverHide(');
    });

    it('any click, scroll or key press clears every hover card', () => {
        const all = between('function hideAllHoverCards(', '// ── END HOVER CARDS');
        for (const id of ['pageBoxTooltip', 'tabTooltip', 'dagTooltip', 'dagTabTooltip']) {
            expect(all, id).toContain(id);
        }
        for (const event of ['mousedown', 'wheel', 'keydown']) {
            expect(html).toContain(`document.addEventListener('${event}', hideAllHoverCards, true);`);
        }
    });
});
