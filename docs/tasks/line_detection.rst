Line Detection
==============

Both line detectors consume an image or crop and produce ``ImageCrop`` artefacts
for text lines. Their only task command is ``run``; neither exposes ``train``.

LineDetectorRLSA
----------------

Detects printed-text lines using image processing, with optional deskewing and
rule removal. It does not download model weights.

.. code-block:: yaml

    detect_lines:
      type: collectra.LineDetectorRLSA
      input: page
      output: lines
      deskew: true

Commands::

    collectra --pipeline pipeline.yaml task detect_lines run images --output results
    collectra --pipeline pipeline.yaml task detect_lines run --help

Configure algorithm parameters in YAML. See :doc:`../quickstart` for a complete
pipeline, and :doc:`../artefacts/image_crop` to extract or evaluate the lines.

LineDetectorSurya
-----------------

Uses the Surya text-detection model. Requires the Surya backend and its weights.

.. code-block:: yaml

    detect_lines:
      type: collectra.LineDetectorSurya
      input: page
      output: lines
      merge_horizontal: false
      min_height: 0

Commands::

    collectra --pipeline pipeline.yaml task detect_lines run images --output results
    collectra --pipeline pipeline.yaml task detect_lines run --help

Both detectors accept the :doc:`shared run options <index>`.
