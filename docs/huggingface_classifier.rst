Hugging Face image classification
================================

``ImageClassifierHuggingFace`` fine-tunes Transformers image classifiers from
labelled Collectra images or crops. It uses ``AutoImageProcessor`` and
``AutoModelForImageClassification`` from the `Hugging Face image classification
guide <https://huggingface.co/docs/transformers/en/tasks/image_classification>`_.
Inference returns a ``Link`` from the predicted class to the input image, matching
the other Collectra classifiers.

Configure the task with an existing image or crop node as its input::

    specimen_classifier:
      type: collectra.ImageClassifierHuggingFace
      input: [specimen]
      output: [Pollen, Spore]
      model: google/vit-base-patch16-224-in21k

Train from annotated Collectra data directories::

    collectra --pipeline pipeline.yaml task specimen_classifier train ./labelled-data \
        --model google/vit-base-patch16-224-in21k --validation validation \
        --epochs 20 --batch 16 --output ./classifier-training

Model loading and classification labels
--------------------------------------

``--model`` accepts:

* A Hugging Face Hub image-classification model ID supported by the installed
  Transformers version, such as ``google/vit-base-patch16-224-in21k``.
* A local ``save_pretrained`` directory containing model weights, model
  configuration, and image-processor configuration.
* A ``.pt`` checkpoint produced by ``ImageClassifierHuggingFace``.

Omitting ``--model`` keeps the task's configured model. A new task without an
explicit model uses ``google/vit-base-patch16-224-in21k``. Hub models load
pretrained weights by default. ``--no-pretrained`` uses a Hub model's configuration
and processor but initializes random weights; local model directories and
Collectra checkpoints always retain their saved weights.

The task configures ``num_labels``, ``id2label``, and ``label2id`` from the sorted
classifier class names. When those names change, it initializes a new
classification head while preserving the backbone, including when the old and
new class counts happen to match. Models must expose a separate Transformers
base model to adapt or freeze their classification head. Built-in Transformers
architectures are supported; custom Hub code is not enabled.

Image preparation and training
------------------------------

Images and crops are exported as oriented RGB PNGs in ``train/<class>`` and
``val/<class>``. ``--validation`` identifies the held-out partition and
``--exclude`` omits another partition. Every configured class needs training
images, but validation may omit individual classes. Use a fresh output directory
for each training run.

The model's image processor performs resizing, cropping, rescaling, and
normalization as configured for that model. Training additionally applies
horizontal flips with probability ``--fliplr`` (default 0.5); set it to 0 if
flipping is inappropriate. ``--min-size`` skips undersized images or crops.

Training uses AdamW, with ``--learning-rate`` and ``--weight-decay`` controlling
optimization. ``--freeze-backbone`` trains the classification head only.
``--device`` selects a Torch device (such as ``cpu``, ``cuda:0``, or ``mps``), and
``--workers`` controls data-loading processes. All parameters have descriptions
in ``train --help``.

The best checkpoint is selected by validation loss. ``--early-stop`` limits the
number of epochs without improvement; 0 disables early stopping. If validation
is omitted, all non-excluded images are used for training, best-model selection
uses training loss, and early stopping is disabled. A specified validation
partition with no usable images is an error.

Saved outputs and reloading
---------------------------

The training directory contains:

* ``train/`` and ``val/``: exported images organized by class.
* ``weights/best.pt`` and ``weights/last.pt``: Collectra checkpoints containing
  weights, model configuration, image-processor settings, and class names.
* ``weights/classes.json``: ordered class names.
* ``weights/huggingface/``: the best model and image processor in standard
  ``save_pretrained`` format, loadable directly by Transformers.
* ``history.csv``: training and validation loss, top-1 accuracy, and top-5 accuracy
  for each epoch. Accuracies are fractions; top-5 uses all classes if there are
  fewer than five.
* ``metrics.json``: best-epoch metrics and the number of epochs completed.

Training copies the best Collectra checkpoint and class metadata into the
pipeline and updates its model path. ``--no-keep-log`` removes the training
directory afterwards; the pipeline checkpoint remains available. Collectra
checkpoints reload without needing the original Hub repository or local source
directory, including their image preprocessing.

To fine-tune a saved model again, use either the Collectra checkpoint or the
native Hugging Face directory::

    collectra --pipeline pipeline.yaml task specimen_classifier train ./labelled-data \
        --model ./classifier-training/weights/huggingface --validation validation

This starts a new optimization run with a fresh optimizer. A bare
``model.safetensors`` or ``pytorch_model.bin`` file is insufficient without the
model and processor configurations; pass its containing ``save_pretrained``
directory instead.
