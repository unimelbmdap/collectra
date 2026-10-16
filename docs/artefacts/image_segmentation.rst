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

The GUI shows the underlying source image with 50% black outside the mask,
initially zoomed to the region. Zoom out to see and paint beyond its current
boundary. Choose **Brush** to add pixels or **Eraser** to remove them; click the
active tool again to return to panning. Both tools use circular brushes. The
Shift key temporarily activates the brush while the pointer is over the image;
Option (Alt) temporarily activates the eraser. Releasing the key restores the
previous tool. If both keys are held, the eraser takes priority.
The **Size** control sets the diameter in source-image pixels. Hold Shift while
scrolling or using a zoom gesture to increase or decrease it. Normal zooming
changes the brush's apparent screen size while preserving its source-pixel size.

Each completed stroke autosaves to the referenced mask PNG. The mask retains
its original source dimensions and binary 0/1 values, including for rotated
views. Source images and the mask reference in ``results.yaml`` are unchanged.
Errors appear beside the tools and the last saved mask is restored. Erasing the
entire mask is rejected because empty ImageSegmentation masks are invalid.

``save()``, ``extract()``, and encoded image
output use the segmented image. Use PNG output to preserve transparency;
extraction without a suffix defaults to PNG. The inherited ``extract`` CLI
command also exports PNG files. Segmentation evaluation is not implemented.
