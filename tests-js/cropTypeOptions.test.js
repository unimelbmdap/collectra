const {
    isImageCropType,
    imageCropDescendantLabels,
    defaultCropLabel,
    humanizeLabel,
} = require('../collectra/gui/cropTypeOptions.js');

// specimen_sheet -(detect)-> {registration_number_image, specific_epithet_image}
//                specific_epithet_image -(ocr)-> specific_epithet (text)
// Type strings match the real results.yaml: collectra.Image / collectra.ImageCrop / collectra.Text.
const adjacency = new Map([
    ['specimen_sheet', ['detect']],
    ['detect', ['registration_number_image', 'specific_epithet_image']],
    ['specific_epithet_image', ['ocr']],
    ['ocr', ['specific_epithet']],
]);
const types = {
    specimen_sheet: 'collectra.Image',
    detect: 'collectra.ObjectDetectionYOLO',
    registration_number_image: 'collectra.ImageCrop',
    specific_epithet_image: 'collectra.ImageCrop',
    ocr: 'collectra.LLM',
    specific_epithet: 'collectra.Text',
};
const typeOf = (label) => types[label] || '';
const isTaskNode = (label) => label === 'detect' || label === 'ocr';

describe('isImageCropType', () => {
    it('matches collectra.ImageCrop', () => {
        expect(isImageCropType('collectra.ImageCrop')).toBe(true);
    });

    it('does not match a plain image type', () => {
        expect(isImageCropType('collectra.Image')).toBe(false);
    });

    it('is false for empty / missing', () => {
        expect(isImageCropType('')).toBe(false);
        expect(isImageCropType(undefined)).toBe(false);
    });
});

describe('imageCropDescendantLabels', () => {
    it('returns ImageCrop-type nodes one data-generation downstream, walking through task nodes', () => {
        expect(imageCropDescendantLabels('specimen_sheet', adjacency, typeOf, isTaskNode))
            .toEqual(['registration_number_image', 'specific_epithet_image']);
    });

    it('excludes the focused node even when it is an ImageCrop', () => {
        expect(imageCropDescendantLabels('specific_epithet_image', adjacency, typeOf, isTaskNode)).toEqual([]);
    });

    it('returns [] for a label with no outgoing edges', () => {
        expect(imageCropDescendantLabels('specific_epithet', adjacency, typeOf, isTaskNode)).toEqual([]);
    });

    it('filters by the type the resolver returns', () => {
        const adj = new Map([['img', ['a', 'b']]]);
        const resolved = { img: 'collectra.Image', a: 'collectra.ImageCrop', b: 'collectra.ImageCrop' };
        expect(imageCropDescendantLabels('img', adj, (l) => resolved[l] || '', () => false)).toEqual(['a', 'b']);
    });

    it('visits a node reachable by two task paths only once', () => {
        const adj = new Map([
            ['img', ['a', 'b']],
            ['a', ['crop']],
            ['b', ['crop']],
        ]);
        const type = (l) => (l === 'crop' ? 'collectra.ImageCrop' : 'collectra.LLM');
        const isTask = (l) => l === 'a' || l === 'b';
        expect(imageCropDescendantLabels('img', adj, type, isTask)).toEqual(['crop']);
    });

    it("does not leak a crop's own further descendants into an ancestor's options", () => {
        // img -(detect)-> crop1 -(analyze)-> crop2 — crop1 is a direct data
        // child of img; crop2 is crop1's, not img's.
        const adj = new Map([
            ['img', ['detect']],
            ['detect', ['crop1']],
            ['crop1', ['analyze']],
            ['analyze', ['crop2']],
        ]);
        const type = (l) => ({
            img: 'collectra.Image',
            detect: 'collectra.ObjectDetectionYOLO',
            crop1: 'collectra.ImageCrop',
            analyze: 'collectra.LLM',
            crop2: 'collectra.ImageCrop',
        }[l] || '');
        const isTask = (l) => l === 'detect' || l === 'analyze';
        expect(imageCropDescendantLabels('img', adj, type, isTask)).toEqual(['crop1']);
    });
});

describe('defaultCropLabel', () => {
    const labels = ['primary_label_unoriented', 'handwritten_data'];

    it('is the first label in pipeline order', () => {
        expect(defaultCropLabel(labels)).toBe('primary_label_unoriented');
    });

    it('stays on the first label even when the page already has an instance of it', () => {
        expect(defaultCropLabel(labels, new Map([['primary_label_unoriented', ['primary_label_unoriented1']]])))
            .toBe('primary_label_unoriented');
    });

    it('returns "" for an empty label list', () => {
        expect(defaultCropLabel([])).toBe('');
    });
});

describe('humanizeLabel', () => {
    it('title-cases an underscore-separated pipeline key', () => {
        expect(humanizeLabel('specific_epithet_image')).toBe('Specific Epithet Image');
    });

    it('handles a single word', () => {
        expect(humanizeLabel('specimen')).toBe('Specimen');
    });

    it('returns "" for a falsy input', () => {
        expect(humanizeLabel('')).toBe('');
    });
});
