// Ports collectra.types.images.Orientation / ImageCrop.set_rel_to_src_parent()
// (the Python package that produces these results.yaml files — see
// ~/projects/mdap/collectra/collectra/types/images.py) so a rotated crop's
// children land in the right place without re-deriving the maths from scratch.
//
// A crop's own x_center/y_center/width_relative/height_relative are always
// stored relative to the RAW, unrotated source image. Displaying a crop
// rotated therefore isn't just "rotate the pixels" — every child nested
// inside it needs its box remapped into that rotated frame too.

// Degrees to rotate a north-up image by to correct it, per Orientation.to_degree().
// Negative = clockwise (matches PIL's Image.rotate(angle, expand=True) convention,
// which this value is fed into directly on the Python side).
const ORIENTATION_DEGREES = { north: 0, west: -90, south: -180, east: -270 };

function orientationDegrees(orientation) {
    const key = (orientation || 'north').toLowerCase();
    return Object.prototype.hasOwnProperty.call(ORIENTATION_DEGREES, key) ? ORIENTATION_DEGREES[key] : 0;
}

// child/parent: { x_center, y_center, width_relative, height_relative }, both
// relative to the same unrotated source image (parent need not be an actual
// crop — pass { x_center: 0.5, y_center: 0.5, width_relative: 1, height_relative: 1 }
// for "parent is the whole page", which is also what makes the step below a
// no-op for a crop's direct children, same as the Python source's own
// `if type(self.source_parent) == ImageCrop` guard).
//
// parentDegrees: the PARENT's own orientationDegrees() — rotating the parent
// is what requires remapping the child, not the child's own orientation.
//
// Returns the child's box as a 0..1 fraction of the parent's own displayed
// (rotated) image.
function rotateChildIntoParentFrame(child, parent, parentDegrees) {
    const dx = child.x_center - parent.x_center;
    const dy = child.y_center - parent.y_center;
    let x = 0.5 + dx / parent.width_relative;
    let y = 0.5 + dy / parent.height_relative;
    let w = child.width_relative / parent.width_relative;
    let h = child.height_relative / parent.height_relative;

    switch (parentDegrees) {
        case -90:
            [x, y] = [1 - y, x];
            [w, h] = [h, w];
            break;
        case -180:
            [x, y] = [1 - x, 1 - y];
            break;
        case -270:
            [x, y] = [y, 1 - x];
            [w, h] = [h, w];
            break;
        // 0 (north): no change
    }
    return { x_center: x, y_center: y, width_relative: w, height_relative: h };
}

// Inverse of rotateChildIntoParentFrame — given a box already expressed as a
// fraction of the parent's rotated frame (e.g. a box just drawn while viewing
// a rotated crop), returns it relative to the same unrotated source image the
// parent's own x_center/y_center/width_relative/height_relative are in. Needed
// to convert a newly-drawn annotation back to what the backend expects.
function unrotateChildFromParentFrame(childInParentFrame, parent, parentDegrees) {
    let x = childInParentFrame.x_center;
    let y = childInParentFrame.y_center;
    let w = childInParentFrame.width_relative;
    let h = childInParentFrame.height_relative;

    switch (parentDegrees) {
        case -90:
            // forward: [x,y] = [1-y, x] ,[w,h] = [h,w] -- invert:
            [x, y] = [y, 1 - x];
            [w, h] = [h, w];
            break;
        case -180:
            [x, y] = [1 - x, 1 - y];
            break;
        case -270:
            [x, y] = [1 - y, x];
            [w, h] = [h, w];
            break;
    }

    return {
        x_center: parent.x_center + (x - 0.5) * parent.width_relative,
        y_center: parent.y_center + (y - 0.5) * parent.height_relative,
        width_relative: w * parent.width_relative,
        height_relative: h * parent.height_relative,
    };
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { ORIENTATION_DEGREES, orientationDegrees, rotateChildIntoParentFrame, unrotateChildFromParentFrame };
}
