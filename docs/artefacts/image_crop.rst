ImageCrop
=========

``collectra.ImageCrop`` represents a region of an image. It stores the source
image path, parent reference, and normalized ``x_center``, ``y_center``,
``width_relative``, and ``height_relative`` coordinates. Detection and
line-detection tasks produce crops; OCR and classifiers can consume them.

Commands
--------

* ``extract INPUTS...``: export the cropped, oriented regions as JPEG files.
  ``--output`` defaults to ``extracted``.
* ``evaluate PREDICTED_FOLDER GOLD_FOLDER``: compare crop bounding boxes using
  intersection-over-union, with the :doc:`shared evaluation options <index>`.

For a crop node named ``lines``:

.. code-block:: bash

    collectra --pipeline pipeline.yaml artefact lines extract results/*.collectra --output extracted-lines
    collectra --pipeline pipeline.yaml artefact lines evaluate predictions gold --threshold 0.5 --output line-metrics.csv
    collectra --pipeline pipeline.yaml artefact lines extract --help
    collectra --pipeline pipeline.yaml artefact lines evaluate --help

Pass result directories containing the relevant artefacts, not raw image files.
The :doc:`GUI <../gui>` can display and edit boxes over their source image.
