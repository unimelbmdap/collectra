ObjectDetectionDETR
===================

Uses a Transformers DETR model to detect objects and return labelled image crops.

.. code-block:: yaml

    detect_objects:
      type: collectra.ObjectDetectionDETR
      model: default
      input: image
      output: [person]

Choose output labels appropriate to your checkpoint or training annotations.

Commands
--------

* ``run INPUTS...``: run the configured detector with the shared run options.
* ``train INPUTS...``: train using ``--model-name`` to select the Transformers
  model. Also supports ``--epochs``, ``--batch``, ``--workers``,
  ``--learning-rate``, ``--weight-decay``, and ``--no-pretrained``.

.. code-block:: bash

    collectra --pipeline pipeline.yaml task detect_objects run images --output results
    collectra --pipeline pipeline.yaml task detect_objects train training-data --model-name facebook/detr-resnet-50 --validation validation
    collectra --pipeline pipeline.yaml task detect_objects train --help
