const { resolveDagClickTarget } = require('../collectra/gui/dagNodeResolution.js');

describe('resolveDagClickTarget', () => {
    it('falls back to the label itself when pageNodeIds has no entry for it', () => {
        const pageNodeIds = new Map();
        const result = resolveDagClickTarget('sandglass', pageNodeIds);
        expect(result).toEqual({ nodeId: 'sandglass', instances: ['sandglass'] });
    });

    it('falls back to the label itself when pageNodeIds maps it to an empty array', () => {
        const pageNodeIds = new Map([['sandglass', []]]);
        const result = resolveDagClickTarget('sandglass', pageNodeIds);
        expect(result).toEqual({ nodeId: 'sandglass', instances: ['sandglass'] });
    });

    it('resolves to the single real id when the label has exactly one instance', () => {
        const pageNodeIds = new Map([['sandglass', ['sandglass1']]]);
        const result = resolveDagClickTarget('sandglass', pageNodeIds);
        expect(result).toEqual({ nodeId: 'sandglass1', instances: ['sandglass1'] });
    });

    it('returns a null nodeId and the full instance list when the label has multiple instances', () => {
        const pageNodeIds = new Map([['sandglass', ['sandglass1', 'sandglass2']]]);
        const result = resolveDagClickTarget('sandglass', pageNodeIds);
        expect(result).toEqual({ nodeId: null, instances: ['sandglass1', 'sandglass2'] });
    });

    it('passes an already-real id through unchanged (not a key in pageNodeIds)', () => {
        const pageNodeIds = new Map([['sandglass', ['sandglass1', 'sandglass2']]]);
        const result = resolveDagClickTarget('sandglass1', pageNodeIds);
        expect(result).toEqual({ nodeId: 'sandglass1', instances: ['sandglass1'] });
    });
});