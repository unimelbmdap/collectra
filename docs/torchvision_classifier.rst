ImageClassifierTorchvision
==========================

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

Commands
--------

* ``run INPUTS...``: classify images or crops using the configured model and
  the :doc:`shared run options <tasks/index>`.
* ``train INPUTS...``: train on labelled Collectra result directories using
  the options described below. Add ``--wandb`` to enable logging; it is off
  by default.

.. code-block:: bash

    collectra --pipeline pipeline.yaml task specimen_classifier run images --output results
    collectra --pipeline pipeline.yaml task specimen_classifier run --help
    collectra --pipeline pipeline.yaml task specimen_classifier train --help

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
``--flipud`` similarly controls vertical flips during training (default 0.0).
Neither flip augmentation is applied during validation or inference.

By default all model parameters are fine-tuned using AdamW. ``--freeze-backbone``
trains only the final projection and freezes the rest of the model, including
batch-normalization statistics. With RGB-view fusion enabled, it trains the
attention modules and final projection while freezing the shared encoder.
``--learning-rate``, ``--weight-decay``,
``--early-stop``, ``--device``, and ``--workers`` control training. Each CLI option
has a description in ``train --help``.

Exported images and results
---------------------------

The output directory contains:

* ``train/<class>`` and ``val/<class>``: extracted images, with crop boundaries and
  orientation applied. TIFF inputs are exported as TIFFs preserving every channel
  and their original dtype; other inputs retain the existing RGB PNG export.
  Filenames are unique within the export.
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

Multichannel TIFF data
----------------------------

TIFF images with more than three channels bypass Pillow throughout export,
training, validation, and inference. TIFF spatial and channel axes are interpreted
from metadata, including channel-first, channel-last, and unlabelled page-stack
layouts. Crops and orientation affect only the spatial axes. Exported TIFFs use
``CYX`` axes and preserve the source dtype and pixel values.

The dataset loader produces float32 ``(C, H, W)`` tensors. Unsigned integer values
are scaled by their dtype maximum (255 for uint8, 65535 for uint16); floating-point
and signed integer values are cast to float32 without rescaling. Resizing, center
cropping, and optional flips preserve all channels.

For channel counts divisible by three, RGB mean/std statistics repeat in channel
order, treating the input as stacked RGB views. Other channel counts use identity
normalization when only RGB statistics are available. Matching per-channel
statistics, or one shared mean/std value, are applied when present. The actual
channel count and normalization are recorded in the checkpoint. Ordinary RGB
inputs retain their existing preprocessing.

All exported training and validation images must have the same channel count.
Mixed counts produce an error listing the affected filenames and counts.
By default, supported ResNet and ConvNeXt models adapt the first convolution by
repeating and averaging its RGB weights across the views. The subsequent model
processes the combined features normally. This remains the default with
``fusion: false``.

Attention fusion of RGB views
-------------------------------

Set the task's ``fusion`` attribute to ``true`` to combine views after shared
feature extraction instead of combining channels in the first convolution:

.. code-block:: yaml

    specimen_classifier:
      type: collectra.ImageClassifierTorchvision
      input: [specimen]
      output: [Pollen, Spore]
      model: resnet18
      fusion: true

Input channels must be ordered ``R1,G1,B1,R2,G2,B2,...`` and their count must be a
positive multiple of three. The number of views is inferred from the dataset:
15 channels means five RGB views; 75 channels means 25. All views receive the
same spatial crop and augmentation, and each RGB triplet uses the model's RGB
normalization statistics.

The views are reshaped into a single batch and passed through **one shared RGB
encoder**. Learned, reference-conditioned softmax attention combines their
feature vectors immediately before the final linear classification projection.
The input convolution stays three-channel; encoder weights are shared across
views. A residual gate starts at zero, preserving the first RGB view's prediction
in evaluation mode initially, and learns how much the other views contribute.

This follows the view-attention approach used by Collectra's YOLO and RF-DETR
fusion models. Those detectors fuse spatial features before their detection
heads; the classifier fuses feature vectors before its final class prediction.
Training cost grows with the number of views, so choose batch size accordingly.

Fusion supports torchvision classifiers with a final linear projection,
including ResNet and ConvNeXt. Convolutional-only heads such as SqueezeNet's are
not supported in fusion mode. Start from an RGB architecture/checkpoint; a
checkpoint whose first convolution was already expanded to multiple views
cannot be converted to this architecture by toggling the flag.

Saved checkpoints contain the view count, attention configuration and all fusion
weights. Loading a fusion checkpoint reconstructs it automatically, even when
``fusion`` is omitted from the task YAML. Fine-tuning preserves its attention
weights; changing classes replaces only the final projection. A dataset with a
different view count is rejected rather than silently discarding learned views.
