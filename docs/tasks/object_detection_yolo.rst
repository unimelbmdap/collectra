ObjectDetectionYOLO
===================

Uses Ultralytics YOLO weights to detect objects and return labelled image crops.
For example, configure a detector trained to identify specimen labels:

.. code-block:: yaml

    detect_labels:
      type: collectra.ObjectDetectionYOLO
      model: models/labels.pt
      input: image
      output: [label]

Replace the checkpoint path and output labels with those for your model.

Commands
--------

* ``run INPUTS...``: detect objects. In addition to the shared run options,
  ``--model`` overrides the checkpoint and ``--imgsz`` sets the inference size.
* ``train INPUTS...``: train on annotated result directories. Supports model,
  epoch, batch, image-size, optimizer, augmentation, device, and checkpoint options.

.. code-block:: bash

    collectra --pipeline pipeline.yaml task detect_labels run images --output results
    collectra --pipeline pipeline.yaml task detect_labels run images --model models/other.pt --imgsz 1280
    collectra --pipeline pipeline.yaml task detect_labels train training-data --epochs 20 --validation validation
    collectra --pipeline pipeline.yaml task detect_labels train --help

Model-specific options and their defaults are described in ``run --help`` and
``train --help``.
