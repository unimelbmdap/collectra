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
``--flipud`` similarly controls vertical flips during training (default 0.0).
Neither flip augmentation is applied during validation or inference.

By default all model parameters are fine-tuned using AdamW. ``--freeze-backbone``
trains only the final projection and freezes the rest of the model, including
batch-normalization statistics. ``--learning-rate``, ``--weight-decay``,
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
---------------------

TIFF images with more than three channels bypass Pillow throughout export,
training, validation, and inference. TIFF spatial and channel axes are interpreted
from metadata, including channel-first, channel-last, and unlabelled page-stack
layouts. Crops and orientation affect only the spatial axes. Exported TIFFs use
``CYX`` axes and preserve the source dtype and pixel values.

The dataset loader produces float32 ``(C, H, W)`` tensors. Unsigned integer values
are scaled by their dtype maximum (255 for uint8, 65535 for uint16); floating-point
and signed integer values are cast to float32 without rescaling. Resizing, center
cropping, and optional flips preserve all channels.

RGB mean/std statistics are not repeated across spectral bands. When a model's
preprocessing contains three-channel statistics but the dataset has a different
channel count, training uses identity normalization and records the channel count
and normalization in the checkpoint. Matching per-channel statistics, or one
shared mean/std value, are applied when present in the preprocessing configuration.
Ordinary RGB inputs retain their existing preprocessing.

All exported training and validation images must have the same channel count.
Mixed counts produce an error listing the affected filenames and counts.
The model and its checkpoint reconstruction must already support the input channel
count; this data-handling support does not adapt model input layers.
