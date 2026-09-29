Image Orientation
=================

ImageOrienter
-------------

``collectra.ImageOrienter`` consumes an image and returns it with a predicted
orientation. Supply a compatible trained orientation-model checkpoint.

.. code-block:: yaml

    orient:
      type: collectra.ImageOrienter
      model: models/orientation.pt
      input: image
      output: oriented_image

Commands
--------

``run`` is the only task command, with the :doc:`shared run options <index>`.
This task does not expose a ``train`` command.

.. code-block:: bash

    collectra --pipeline pipeline.yaml task orient run images --output results
    collectra --pipeline pipeline.yaml task orient run --help
