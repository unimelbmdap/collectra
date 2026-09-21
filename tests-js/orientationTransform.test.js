const {
    ORIENTATION_DEGREES,
    orientationDegrees,
    rotateChildIntoParentFrame,
    unrotateChildFromParentFrame,
} = require('../collectra/gui/orientationTransform.js');

const WHOLE_PAGE = { x_center: 0.5, y_center: 0.5, width_relative: 1, height_relative: 1 };

describe('orientationDegrees', () => {
    it('matches collectra.types.images.Orientation.to_degree()', () => {
        expect(orientationDegrees('north')).toBe(0);
        expect(orientationDegrees('west')).toBe(-90);
        expect(orientationDegrees('south')).toBe(-180);
        expect(orientationDegrees('east')).toBe(-270);
    });

    it('is case-insensitive', () => {
        expect(orientationDegrees('WEST')).toBe(-90);
    });

    it('defaults to 0 for missing/unknown values', () => {
        expect(orientationDegrees('')).toBe(0);
        expect(orientationDegrees(undefined)).toBe(0);
        expect(orientationDegrees('sideways')).toBe(0);
    });

    it('exposes the raw table', () => {
        expect(ORIENTATION_DEGREES).toEqual({ north: 0, west: -90, south: -180, east: -270 });
    });
});

describe('rotateChildIntoParentFrame', () => {
    it('is a no-op when the parent is north (0deg), whole page', () => {
        const child = { x_center: 0.75, y_center: 0.25, width_relative: 0.5, height_relative: 0.5 };
        expect(rotateChildIntoParentFrame(child, WHOLE_PAGE, 0)).toEqual(child);
    });

    it('rotates a top-right child to the bottom-right when the page is west (-90deg)', () => {
        // A standard 90deg-clockwise point rotation about the centre: (x,y) -> (1-y, x).
        const child = { x_center: 0.75, y_center: 0.25, width_relative: 0.5, height_relative: 0.5 };
        expect(rotateChildIntoParentFrame(child, WHOLE_PAGE, -90))
            .toEqual({ x_center: 0.75, y_center: 0.75, width_relative: 0.5, height_relative: 0.5 });
    });

    it('rotates a top-right child to the bottom-left when the page is south (-180deg)', () => {
        const child = { x_center: 0.75, y_center: 0.25, width_relative: 0.5, height_relative: 0.5 };
        expect(rotateChildIntoParentFrame(child, WHOLE_PAGE, -180))
            .toEqual({ x_center: 0.25, y_center: 0.75, width_relative: 0.5, height_relative: 0.5 });
    });

    it('rotates a top-right child to the top-left when the page is east (-270deg)', () => {
        const child = { x_center: 0.75, y_center: 0.25, width_relative: 0.5, height_relative: 0.5 };
        expect(rotateChildIntoParentFrame(child, WHOLE_PAGE, -270))
            .toEqual({ x_center: 0.25, y_center: 0.25, width_relative: 0.5, height_relative: 0.5 });
    });

    it('swaps width/height on a +-90deg rotation but not on 180deg', () => {
        const child = { x_center: 0.5, y_center: 0.5, width_relative: 0.4, height_relative: 0.2 };
        expect(rotateChildIntoParentFrame(child, WHOLE_PAGE, -90).width_relative).toBe(0.2);
        expect(rotateChildIntoParentFrame(child, WHOLE_PAGE, -90).height_relative).toBe(0.4);
        expect(rotateChildIntoParentFrame(child, WHOLE_PAGE, -270).width_relative).toBe(0.2);
        expect(rotateChildIntoParentFrame(child, WHOLE_PAGE, -180).width_relative).toBe(0.4);
    });

    it('normalises a child nested inside an actual (unrotated) crop, not just the whole page', () => {
        // parent crop spans x:[0.4,0.6] y:[0.4,0.6]; child spans x:[0.45,0.55] y:[0.4,0.5]
        // -> in the parent's own 0..1 frame that's centred at (0.5, 0.25) sized 0.5x0.5
        const parent = { x_center: 0.5, y_center: 0.5, width_relative: 0.2, height_relative: 0.2 };
        const child = { x_center: 0.5, y_center: 0.45, width_relative: 0.1, height_relative: 0.1 };
        const result = rotateChildIntoParentFrame(child, parent, 0);
        expect(result.x_center).toBeCloseTo(0.5, 10);
        expect(result.y_center).toBeCloseTo(0.25, 10);
        expect(result.width_relative).toBeCloseTo(0.5, 10);
        expect(result.height_relative).toBeCloseTo(0.5, 10);
    });
});

describe('unrotateChildFromParentFrame is the exact inverse of rotateChildIntoParentFrame', () => {
    const cases = [0, -90, -180, -270];
    const parents = [
        WHOLE_PAGE,
        { x_center: 0.3, y_center: 0.6, width_relative: 0.25, height_relative: 0.4 },
    ];
    const child = { x_center: 0.34, y_center: 0.58, width_relative: 0.05, height_relative: 0.08 };

    for (const degrees of cases) {
        for (const parent of parents) {
            it(`round-trips at ${degrees}deg for parent ${JSON.stringify(parent)}`, () => {
                const rotated = rotateChildIntoParentFrame(child, parent, degrees);
                const back = unrotateChildFromParentFrame(rotated, parent, degrees);
                expect(back.x_center).toBeCloseTo(child.x_center, 10);
                expect(back.y_center).toBeCloseTo(child.y_center, 10);
                expect(back.width_relative).toBeCloseTo(child.width_relative, 10);
                expect(back.height_relative).toBeCloseTo(child.height_relative, 10);
            });
        }
    }
});
