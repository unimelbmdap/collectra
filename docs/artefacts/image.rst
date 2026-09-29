Image
=====

``collectra.Image`` represents a source image, with a file path, orientation,
and parent relationships. A serialized image in ``results.yaml`` looks like:

.. code-block:: yaml

    page:
      type: collectra.Image
      id: page-001
      data: page001.png

Relative data paths resolve from the results directory. Pipeline root image
nodes can be inferred from task inputs; do not instantiate an Image with an
empty file path in the pipeline configuration.

Commands
--------

``extract INPUTS...`` exports the node's images as JPEG files. ``--output``
defaults to ``extracted``. Use the node name (``page`` here):

.. code-block:: bash

    collectra --pipeline pipeline.yaml artefact page extract results/page001.collectra --output extracted-pages
    collectra --pipeline pipeline.yaml artefact page extract --help

The inherited ``evaluate`` command appears in help, but whole ``Image``
artefacts do not implement an evaluation score. Use :doc:`image_crop` for box
evaluation, or :doc:`text` for recognized text. There is no image training or
inference command under ``artefact``.
