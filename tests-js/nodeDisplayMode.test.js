const { decideNodeDisplayMode } = require('../collectra/gui/nodeDisplayMode.js');

describe('decideNodeDisplayMode', () => {
    it('routes a node whose id contains "tei" to fileBackedText, regardless of type', () => {
        const mode = decideNodeDisplayMode({ nodeId: 'tei', realType: '', pageViewerActive: true });
        expect(mode).toBe('fileBackedText');
    });

    it('routes a node whose id contains "markdown" to fileBackedText, regardless of type', () => {
        const mode = decideNodeDisplayMode({ nodeId: 'markdown', realType: '', pageViewerActive: true });
        expect(mode).toBe('fileBackedText');
    });

    it('routes a real annotation Text field to annotationText (the formation/genus bug)', () => {
        const mode = decideNodeDisplayMode({
            nodeId: 'formation',
            realType: 'collectra.Text',
            pageViewerActive: true,
        });
        expect(mode).toBe('annotationText');
    });

    it('routes an ImageCrop node to imageHighlight when the page viewer is active', () => {
        const mode = decideNodeDisplayMode({
            nodeId: 'genus_image',
            realType: 'collectra.ImageCrop',
            pageViewerActive: true,
        });
        expect(mode).toBe('imageHighlight');
    });

    it('routes an unknown/implicit type (empty realType) to imageHighlight when the page viewer is active', () => {
        const mode = decideNodeDisplayMode({
            nodeId: 'label_orientator',
            realType: '',
            pageViewerActive: true,
        });
        expect(mode).toBe('imageHighlight');
    });

    it('falls back to legacyNodeDisplay when there is no active page viewer', () => {
        const mode = decideNodeDisplayMode({
            nodeId: 'genus_image',
            realType: 'collectra.ImageCrop',
            pageViewerActive: false,
        });
        expect(mode).toBe('legacyNodeDisplay');
    });

    it('does not treat a task node whose name merely contains "tei" as the file-backed tei node', () => {
        const mode = decideNodeDisplayMode({
            nodeId: 'tei_converter',
            realType: '',
            pageViewerActive: true,
        });
        expect(mode).toBe('imageHighlight');
    });

    it('does not treat a task node whose name merely contains "markdown" as the file-backed markdown node', () => {
        const mode = decideNodeDisplayMode({
            nodeId: 'markdown_converter',
            realType: '',
            pageViewerActive: true,
        });
        expect(mode).toBe('imageHighlight');
    });

    it('a Text-type node still wins over the image/legacy checks when id does not contain tei/markdown', () => {
        const mode = decideNodeDisplayMode({
            nodeId: 'registration_number',
            realType: 'collectra.Text',
            pageViewerActive: true,
        });
        expect(mode).toBe('annotationText');
    });
});
