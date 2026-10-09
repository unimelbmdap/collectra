const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync(require.resolve('../collectra/gui/index.html'), 'utf8');
const start = html.indexOf('async function loadCollectraFolderInner(index)');
const end = html.indexOf('// Collapse the sidebar', start);
const source = html.slice(start, end);

it.each([null, 'crop'])('renders only the final selected artefact when carrying %s', async carry => {
    const item = {classList: {add: vi.fn(), remove: vi.fn()}};
    const elements = {folderInfo: {}, yamlPath: {}};
    const bottom = {
        panel: {pruneTabs: vi.fn()}, renderTabStripView: vi.fn(),
        showActiveTabContent: vi.fn(), renderPageArtefact: vi.fn(), openOrActivateTab: vi.fn(),
    };
    const other = {panel: {pruneTabs: vi.fn()}, renderTabStripView: vi.fn(), showActiveTabContent: vi.fn()};
    const context = {
        document: {querySelectorAll: () => [item], getElementById: id => elements[id]},
        window: {pywebview: {api: {
            load_collectra_folder: vi.fn().mockResolvedValue({success: true, folder_path: '/page', image_path: '/page/image.png', yaml_path: '/page/results.yaml'}),
            get_label_statistics: vi.fn().mockResolvedValue({success: true, label_counts: {}}),
            get_active_node_ids: vi.fn().mockResolvedValue({success: true, active_ids: ['image', 'crop'], node_ids: {crop: ['crop-001']}}),
        }}},
        loadYaml: vi.fn(), updateNavButtons: vi.fn(), populateTabs: vi.fn(), setActive: vi.fn(),
        panelControllers: {bottom, other}, activeNodeIds: new Set(['image', 'crop']),
        resolveDagLabel: id => id, pendingCarryLabel: carry, PAGE_NODE_RE: /^Page/,
        cy: {nodes: () => ({roots: () => [{id: () => 'image'}]})}, lastDagNodes: [],
        showResult: vi.fn(), formatLabelCounts: vi.fn(),
    };
    await vm.runInNewContext(`${source}\nloadCollectraFolderInner(0)`, context);
    expect(bottom.renderPageArtefact).not.toHaveBeenCalled();
    expect(bottom.showActiveTabContent).not.toHaveBeenCalled();
    expect(bottom.openOrActivateTab).toHaveBeenCalledTimes(1);
    expect(bottom.openOrActivateTab).toHaveBeenCalledWith(carry ? 'crop-001' : 'image');
    expect(other.showActiveTabContent).toHaveBeenCalledTimes(1);
    expect(context.showResult).not.toHaveBeenCalled();
});
