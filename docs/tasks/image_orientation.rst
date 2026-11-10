Image Orientation
=================

ImageOrienter
-------------

``collectra.ImageOrienter`` consumes an image and returns it with a predicted
orientation. ``model`` is a compatible orientation-model checkpoint. Without
it, the published model of `deep-image-orientation-detection
<https://github.com/duartebarbosadev/deep-image-orientation-detection>`_ (MIT
licensed) is downloaded from Hugging Face on first use and cached. On machines
without internet access, download it once beforehand or give ``model``.

Object detectors can orient the crops of chosen classes themselves with their
``orient`` setting; see :doc:`object_detection`.

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
