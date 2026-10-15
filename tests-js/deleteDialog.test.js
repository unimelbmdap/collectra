const {deleteDialogContent, shortTypeName, typeChipHtml} = require('../collectra/gui/deleteDialog.js');

const child = (id, type, depth = 1) => ({id, type, label: '', depth});
const target = (id, name = '') => ({id, name});

it('titles the dialog with the id of the item being deleted', () => {
    expect(deleteDialogContent(target('primary_label_unoriented1'), {children: []}).title)
        .toBe('Delete "primary_label_unoriented1"?');
});

it('puts a custom name in front of the id when there is one', () => {
    expect(deleteDialogContent(target('stamp1', 'Red stamp'), {children: []}).title)
        .toBe('Delete "Red stamp" (stamp1)?');
});

it('does not repeat the id when the name is the id', () => {
    expect(deleteDialogContent(target('stamp1', 'stamp1'), {children: []}).title)
        .toBe('Delete "stamp1"?');
});

it('has no summary or items when nothing is nested inside', () => {
    const content = deleteDialogContent(target('stamp1'), {children: []});
    expect(content.summary).toBe('');
    expect(content.items).toEqual([]);
});

it('says how many descendants go with the item, singular and plural', () => {
    const one = deleteDialogContent(target('a'), {children: [child('t1', 'collectra.Text')]});
    const three = deleteDialogContent(target('a'), {children: [
        child('c1', 'collectra.ImageCrop'), child('t1', 'collectra.Text'), child('t2', 'collectra.Text'),
    ]});
    expect(one.summary).toBe('This will also delete 1 descendant:');
    expect(three.summary).toBe('This will also delete 3 descendants:');
});

it('lists each descendant by short type, id and depth, in the given order', () => {
    const content = deleteDialogContent(target('a'), {children: [
        child('c1', 'collectra.ImageCrop', 1),
        child('t1', 'collectra.Text', 2),
    ]});
    expect(content.items).toEqual([
        {type: 'ImageCrop', id: 'c1', depth: 1},
        {type: 'Text', id: 't1', depth: 2},
    ]);
});

it('defaults a missing depth to the first level', () => {
    const content = deleteDialogContent(target('a'), {children: [{id: 'c1', type: 'collectra.Text'}]});
    expect(content.items[0].depth).toBe(1);
});

it('says the item itself is kept when only its contents are deleted', () => {
    const content = deleteDialogContent(target('image'), {keeps_self: true, children: [child('c1', 'collectra.ImageCrop')]});
    expect(content.title).toBe('Delete everything under "image"?');
    expect(content.summary).toBe('This will delete 1 descendant. The image itself is kept:');
    const many = deleteDialogContent(target('image'), {keeps_self: true, children: [child('a', 'x'), child('b', 'x')]});
    expect(many.summary).toBe('This will delete 2 descendants. The image itself is kept:');
});

it('falls back to a plain dialog when the preview is missing or failed', () => {
    expect(deleteDialogContent(target('a'), null).items).toEqual([]);
    expect(deleteDialogContent(target('a'), {success: false, error: 'x'}).items).toEqual([]);
    expect(deleteDialogContent(target('a'), undefined).summary).toBe('');
});

it('page puts the type chip first in a fixed column, then the name nested by depth', () => {
    const html = require('node:fs').readFileSync(require.resolve('../collectra/gui/index.html'), 'utf8');
    expect(html).toContain('row.append(type, id);');
    expect(html).toContain("id.style.paddingLeft = ((item.depth - 1) * 16) + 'px';");
    expect(html).toContain("id.classList.toggle('nested', item.depth > 1);");
    expect(html).toContain('.delete-children .delete-child-id.nested::before');
    expect(html).toContain("content: '\\21B3\\00a0';");
});

it('shortens a type to its plain name for chips', () => {
    expect(shortTypeName('collectra.ImageCrop')).toBe('ImageCrop');
    expect(shortTypeName('Text')).toBe('Text');
    expect(shortTypeName('')).toBe('');
    expect(shortTypeName(undefined)).toBe('');
});

it('uses the same short type for the DAG hover chip', () => {
    const html = require('node:fs').readFileSync(require.resolve('../collectra/gui/index.html'), 'utf8');
    expect(html).toContain('const displayType = shortTypeName(pageNodeTypes.get(data.id) || data.detail);');
});

it('DAG hover shows just the type chip when a node has a type, and the name only when it has none', () => {
    const html = require('node:fs').readFileSync(require.resolve('../collectra/gui/index.html'), 'utf8');
    expect(html).toContain("const title = displayType ? '' : `<strong style=\"font-size:12px\">${data.id}</strong>`;");
    expect(html).toContain('${title}${badge}');
    expect(html).not.toContain('<strong style="font-size:12px">${data.id}</strong>${badge}');
});

describe('typeChipHtml (the chip shared by the DAG hover and the tab hover)', () => {
    it('is the outlined monospace chip with the short type', () => {
        const chip = typeChipHtml('collectra.Text');
        expect(chip).toContain('>Text</span>');
        expect(chip).not.toContain('collectra.');
        expect(chip).toContain('font-family:monospace');
        expect(chip).toContain('border:1px solid currentColor');
    });

    it('is empty when there is no type', () => {
        expect(typeChipHtml('')).toBe('');
        expect(typeChipHtml(undefined)).toBe('');
    });

    it('is used by the DAG hover', () => {
        const html = require('node:fs').readFileSync(require.resolve('../collectra/gui/index.html'), 'utf8');
        const dag = html.slice(html.indexOf('function showDagTooltip('), html.indexOf('function hideDagTooltip('));
        expect(dag).toContain('const badge = typeChipHtml(displayType);');
    });
});
