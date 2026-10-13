// Plain type name for chips: "collectra.ImageCrop" -> "ImageCrop".
function shortTypeName(type) {
    return String(type || '').split('.').pop();
}

// What the delete-confirmation popup shows for an item: its title and the
// nested items that are deleted along with it.
function deleteDialogContent(target, preview) {
    const children = (preview && preview.success !== false && preview.children) || [];
    const keepsSelf = Boolean(preview && preview.keeps_self);
    const count = children.length;
    const noun = count === 1 ? 'descendant' : 'descendants';
    const label = target.name && target.name !== target.id ? `${target.name}" (${target.id})` : `${target.id}"`;
    let summary = '';
    if (count) {
        summary = keepsSelf
            ? `This will delete ${count} ${noun}. The image itself is kept:`
            : `This will also delete ${count} ${noun}:`;
    }
    return {
        title: keepsSelf ? `Delete everything under "${target.id}"?` : `Delete "${label}?`,
        summary,
        items: children.map((child) => ({
            type: shortTypeName(child.type),
            id: child.id,
            depth: child.depth || 1,
        })),
    };
}

if (typeof module !== 'undefined') module.exports = {deleteDialogContent, shortTypeName};
