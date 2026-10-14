// Runs the page's own startup script against a stand-in browser, so an error
// thrown while the page loads (which stops everything after it) fails a test
// instead of showing up as an empty window.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const gui = path.resolve(__dirname, '../collectra/gui');
const html = fs.readFileSync(path.join(gui, 'index.html'), 'utf8');
const inline = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const localScripts = [...html.matchAll(/<script src="([^"]+)"/g)]
    .map((m) => m[1])
    .filter((src) => !src.startsWith('vendor') && !src.startsWith('openseadragon'));

// Every property is a callable, chainable dummy, so only genuine script errors surface.
function dummy() {
    const fn = function () { return dummy(); };
    return new Proxy(fn, {
        get(target, prop) {
            if (prop === Symbol.toPrimitive) return () => '';
            if (prop === 'then') return undefined;
            if (prop === 'length') return 0;
            if (prop === Symbol.iterator) return function* () {};
            if (prop === 'children' || prop === 'childNodes') return [];
            return dummy();
        },
        set() { return true; },
        apply() { return dummy(); },
        construct() { return dummy(); },
    });
}

function standInBrowser() {
    const context = {
        console: {log() {}, warn() {}, error() {}},
        setTimeout, clearTimeout, setInterval: () => 0, clearInterval() {},
        Promise, Map, Set, URL, JSON, Math, Date, Array, Object, String, Number, RegExp, Error,
        requestAnimationFrame: () => 0, cancelAnimationFrame() {},
        ResizeObserver: class { observe() {} disconnect() {} },
        MutationObserver: class { observe() {} disconnect() {} },
        IntersectionObserver: class { observe() {} disconnect() {} },
        getComputedStyle: () => dummy(), localStorage: dummy(), navigator: {userAgent: 'test'},
        cytoscape: dummy(), OpenSeadragon: dummy(), AnnotoriousOSD: dummy(), EasyMDE: dummy(),
    };
    context.document = new Proxy({addEventListener() {}, removeEventListener() {}}, {
        get: (target, prop) => (prop in target ? target[prop] : dummy()),
    });
    context.window = new Proxy(context, {
        get: (target, prop) => (prop in target ? target[prop] : dummy()),
        set: (target, prop, value) => { target[prop] = value; return true; },
    });
    context.addEventListener = () => {};
    context.globalThis = context.window;
    return vm.createContext(context);
}

it('the page script loads to the end without throwing', () => {
    const context = standInBrowser();
    for (const src of localScripts) {
        vm.runInContext(fs.readFileSync(path.join(gui, src), 'utf8'), context, {filename: src});
    }
    expect(() => vm.runInContext(inline, context, {filename: 'index.html'})).not.toThrow();
});
