ImageSegmentation
=================

``collectra.ImageSegmentation`` inherits from Image and defines a region with a
binary PNG mask. ``data`` references the source image and ``mask`` references a
single-channel PNG with the same width and height. Mask pixels are 1 inside the
region and 0 outside; one-bit PNG masks are also supported.

.. code-block:: yaml

    segment:
      type: collectra.ImageSegmentation
      id: segment-001
      parents: page-001
      data: page.png
      mask: masks/segment-001.png

Paths are relative to the results directory. ``pil()`` returns an RGBA image
cropped to the mask's bounding box, with pixels outside the mask transparent.
Existing source transparency is preserved inside the mask. Orientation is
applied after masking and cropping. Empty masks, nonbinary masks, and masks
whose dimensions differ from the source image raise errors.

GUI previews use the same mask. ``save()``, ``extract()``, and encoded image
output use the segmented image. Use PNG output to preserve transparency;
extraction without a suffix defaults to PNG. The inherited ``extract`` CLI
command also exports PNG files. Segmentation evaluation is not implemented.
