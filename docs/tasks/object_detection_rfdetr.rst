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

See :doc:`../licensing` when choosing checkpoints; RF-DETR Plus components
have different terms from the Apache-designated models.

Multiview feature fusion
-----------------------

For spatially aligned RGB views, ``adapt-rfdetr-feature-fusion.py`` creates a
checkpoint that applies a shared pretrained RGB encoder to each view and learns
attention across views at corresponding feature locations. Fusion runs before
the existing projector and detection decoder. A residual gate starts at zero,
so the initial output preserves the selected RGB reference view.

.. code-block:: bash

    python adapt-rfdetr-feature-fusion.py rf-detr-nano.pth --num-views 5 --variant nano --reference-view 0 --output models/nano-fusion.pth

Configure ``model: models/nano-fusion.pth`` in the task and train through the
usual Collectra command. Input TIFF channels must be ordered
``R1,G1,B1,R2,G2,B2,...`` and all views must be registered. Reference indices are
zero-based. Integer pixels scale by their dtype maximum; float pixels must
already be in ``[0,1]``. Fusion parameters use the main training learning rate,
while the pretrained encoder retains its usual layer-wise learning rates.

Fusion checkpoints require Collectra's fusion loader; ordinary RF-DETR loading
cannot reconstruct this architecture. From Python, use
``load_adapted_model`` and ``save_adapted_model`` from
``collectra.tasks.object_detection.rfdetr_feature_fusion``. Upstream best/EMA
checkpoints retain the fusion specification and can also be loaded this way.
The loader binds training to a reconstruction path that preserves the fusion
modules and current weights. Training uses torchvision augmentation with
``default`` or ``none`` presets and currently supports one device. ONNX export
is not supported. Compute and activation memory increase with the view count.
