const fs = require('node:fs');
const vm = require('node:vm');
const { lineageNeighbors } = require('../collectra/gui/lineage.js');
const html = fs.readFileSync(require.resolve('../collectra/gui/index.html'), 'utf8');
const between = (from, to) => html.slice(html.indexOf(from), html.indexOf(to, html.indexOf(from)));

// sheet(data) -> detect(task) -> crop(data) -> read(task) -> text(data)
//                            \-> hand(data)
const nodes = [
    {id: 'sheet', nodeType: 'data'}, {id: 'detect', nodeType: 'detector'},
    {id: 'crop', nodeType: 'data'}, {id: 'hand', nodeType: 'data'},
    {id: 'read', nodeType: 'llm'}, {id: 'text', nodeType: 'data'},
];
const edges = [
    {source: 'sheet', target: 'detect'}, {source: 'detect', target: 'crop'}, {source: 'detect', target: 'hand'},
    {source: 'crop', target: 'read'}, {source: 'read', target: 'text'},
];

describe('lineageNeighbors', () => {
    it('finds descendants of the same kind, passing through the tasks between them', () => {
        expect(lineageNeighbors('sheet', 'descendants', nodes, edges)).toEqual(['crop', 'hand', 'text']);
    });

    it('finds ancestors of the same kind at any depth', () => {
        expect(lineageNeighbors('text', 'ancestors', nodes, edges)).toEqual(['crop', 'sheet']);
    });

    it('a task sees only tasks', () => {
        expect(lineageNeighbors('read', 'ancestors', nodes, edges)).toEqual(['detect']);
        expect(lineageNeighbors('detect', 'descendants', nodes, edges)).toEqual(['read']);
    });

    it('has none at the ends of the pipeline', () => {
        expect(lineageNeighbors('sheet', 'ancestors', nodes, edges)).toEqual([]);
        expect(lineageNeighbors('text', 'descendants', nodes, edges)).toEqual([]);
        expect(lineageNeighbors('hand', 'descendants', nodes, edges)).toEqual([]);
    });

    it('is empty for a node the pipeline does not have', () => {
        expect(lineageNeighbors('nope', 'ancestors', nodes, edges)).toEqual([]);
        expect(lineageNeighbors(undefined, 'descendants', nodes, edges)).toEqual([]);
    });

    it('skips nodes it is told to, and does not walk through them', () => {
        const skip = (id) => id === 'crop';
        expect(lineageNeighbors('sheet', 'descendants', nodes, edges, skip)).toEqual(['hand']);
    });

    it('lists each node once and survives a cycle', () => {
        const loop = [...edges, {source: 'text', target: 'sheet'}];
        const result = lineageNeighbors('sheet', 'descendants', nodes, loop);
        expect(new Set(result).size).toBe(result.length);
        expect(result).not.toContain('sheet');
    });

    it('copes with no nodes or edges', () => {
        expect(lineageNeighbors('sheet', 'ancestors', [], [])).toEqual([]);
        expect(lineageNeighbors('sheet', 'ancestors', undefined, undefined)).toEqual([]);
    });
});

describe('Ancestors and Descendants do not depend on any panel being open', () => {
    const source = between('function lineageAvailability(', 'function openLineageFor(');

    function page({withGraph}) {
        const context = {
            lineageNeighbors, resolveDagLabel: (id) => id.replace(/\d+$/, ''), PAGE_NODE_RE: /^Page/,
            lastDagNodes: nodes, lastDagEdges: edges,
            // The drawn graph of the top DAG panel: gone (that panel was removed).
            cy: withGraph ? {} : null,
        };
        vm.createContext(context);
        vm.runInContext(source, context);
        return context;
    }

    it('says a node has ancestors and descendants even when there is no drawn graph at all', () => {
        const ctx = page({withGraph: false});
        expect(ctx.lineageAvailability('crop1')).toEqual({ancestors: true, descendants: true});
        expect(ctx.lineageAvailability('sheet')).toEqual({ancestors: false, descendants: true});
        expect(ctx.lineageAvailability('text')).toEqual({ancestors: true, descendants: false});
    });

    it('gives the same answer with or without a drawn graph', () => {
        expect(page({withGraph: true}).lineageAvailability('crop')).toEqual(page({withGraph: false}).lineageAvailability('crop'));
    });

    it('says none for no node or an unknown one', () => {
        const ctx = page({withGraph: false});
        expect(ctx.lineageAvailability(null)).toEqual({ancestors: false, descendants: false});
        expect(ctx.lineageAvailability('unknown_label')).toEqual({ancestors: false, descendants: false});
    });

    it('the lists in the popovers come from the same data, never from the drawn graph', () => {
        const render = between('function renderLineage(', '\n        }\n');
        expect(render).toContain("lineageNeighbors(resolveDagLabel(nodeId), 'ancestors'");
        expect(render).toContain("lineageNeighbors(resolveDagLabel(nodeId), 'descendants'");
        expect(render).not.toContain('cy.');
        expect(render).not.toContain('!cy');
        expect(source).not.toContain('cy.');
    });

    it('loads the module', () => {
        expect(html).toContain('<script src="lineage.js"></script>');
    });

    it('the old walk over the drawn graph is gone', () => {
        expect(html).not.toContain('function sameTypeNeighbors(');
        expect(html).not.toContain('LINEAGE_TASK_TYPES');
    });
});

describe('the buttons catch up when the pipeline data arrives', () => {
    it('every panel\'s row is redrawn as soon as the nodes and edges are known', () => {
        const loaded = between('lastDagEdges = edges;', 'initCytoscape(');
        expect(loaded).toContain('Object.values(panelControllers).forEach((c) => c.renderPager());');
    });
});
