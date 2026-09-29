ObjectDetectionRFDETR
=====================

Uses RF-DETR to detect objects and return labelled image crops. Select a
pretrained variant or a checkpoint trained for your collection:

.. code-block:: yaml

    detect_labels:
      type: collectra.ObjectDetectionRFDETR
      model: models/labels.pth
      input: image
      output: [label]

Commands
--------

* ``run INPUTS...``: run the configured detector with the shared run options.
* ``train INPUTS...``: fine-tune on annotated result directories. ``--model``
  accepts a checkpoint or supported variant such as ``nano``, ``small``,
  ``medium``, or ``large``. ``--resolution``, ``--grad-accum-steps``, augmentation,
  EMA, and early-stopping options are available in ``train --help``.

.. code-block:: bash

    collectra --pipeline pipeline.yaml task detect_labels run images --output results
    collectra --pipeline pipeline.yaml task detect_labels train training-data --model nano --epochs 20 --validation validation
    collectra --pipeline pipeline.yaml task detect_labels run --help
    collectra --pipeline pipeline.yaml task detect_labels train --help

Variant availability depends on the installed RF-DETR package. Some variants
require its optional packages. Add ``--no-wandb`` to disable W&B logging.
