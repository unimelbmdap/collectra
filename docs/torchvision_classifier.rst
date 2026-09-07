Torchvision image classification
================================

``ImageClassifierTorchvision`` trains an image classifier from labelled images or
crops and returns a ``Link`` from the predicted class to the input image.
Configure it in a pipeline using the existing image or crop node as its input::

    specimen_classifier:
      type: collectra.ImageClassifierTorchvision
      input: [specimen]
      output: [Pollen, Spore]
      model: resnet18

Train it from annotated Collectra data directories::

    collectra --pipeline pipeline.yaml task specimen_classifier train ./labelled-data \
        --model efficientnet_b0 --validation validation --epochs 20 --batch 16 \
        --output ./classifier-training

``--validation`` selects the partition value identifying held-out images.
``--exclude`` omits another partition. Every configured class must have training
images; validation may omit individual classes. If ``--validation`` is omitted,
all non-excluded images are training examples, checkpoint selection uses training
loss, and early stopping is disabled. A specified validation partition with no
usable images is an error.

Models and preprocessing
------------------------

``--model`` accepts an architecture name from the installed torchvision
classification registry, such as ``resnet50``, ``convnext_tiny``, ``mobilenet_v3_small``,
``swin_t``, or ``vit_b_16``. See the `torchvision classification models and weights
<https://docs.pytorch.org/vision/main/models.html#classification>`_. Named models
load their ``DEFAULT`` pretrained weights, downloading them when necessary.
``--no-pretrained`` initializes a named model with random weights instead.
Omitting ``--model`` keeps the task's configured model; a newly configured task
without a model uses ``resnet18``.

The final classification projection is replaced to match the sorted class names.
This supports both linear heads and SqueezeNet's convolutional head. Auxiliary
Inception and GoogLeNet heads are disabled. Training and inference use the chosen
weights' resize, crop, interpolation, and normalization settings. Training adds
horizontal flips with probability ``--fliplr`` (default 0.5); use ``--fliplr 0``
when flipping is inappropriate for the images.

By default all model parameters are fine-tuned using AdamW. ``--freeze-backbone``
trains only the final projection and freezes the rest of the model, including
batch-normalization statistics. ``--learning-rate``, ``--weight-decay``,
``--early-stop``, ``--device``, and ``--workers`` control training. Each CLI option
has a description in ``train --help``.

Exported images and results
---------------------------

The output directory contains:

* ``train/<class>/*.png`` and ``val/<class>/*.png``: extracted RGB images, with crop
  boundaries and orientation applied. Filenames are unique within the export.
* ``weights/best.pt``: the checkpoint with the lowest validation loss (or training
  loss when validation is not requested).
* ``weights/last.pt``: the last completed epoch's checkpoint.
* ``weights/classes.json``: the ordered class names.
* ``history.csv``: loss, top-1 accuracy, and top-5 accuracy for each epoch.
* ``metrics.json``: best-epoch metrics and the number of completed epochs.

Accuracies are fractions from 0 to 1. Top-5 uses all classes when there are fewer
than five. ``--min-size`` filters out images or crops below a pixel threshold.
Empty crops are skipped. Use a fresh output directory for each run to avoid mixing
old exports into a new dataset.

Training copies the best model and class metadata into the pipeline and updates
its configured model path. ``--no-keep-log`` removes the training directory after
that copy; the pipeline's model remains available for inference.

Loading a checkpoint
--------------------

Pass a checkpoint produced by this classifier to fine-tune it again::

    collectra --pipeline pipeline.yaml task specimen_classifier train ./labelled-data \
        --model ./classifier-training/weights/best.pt --validation validation

Checkpoints contain the architecture, model state, class order, and preprocessing
configuration, so reloading does not download pretrained weights. An unchanged
class list preserves the trained head; a different class list creates a new head.
This starts a new fine-tuning run with a fresh optimizer, rather than resuming the
optimizer state of a previous run.

Bare external state dictionaries are not sufficient on their own: they do not
identify the architecture, class labels, and preprocessing. The loader reports
an error for checkpoints missing this metadata.
