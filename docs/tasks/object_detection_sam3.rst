ObjectDetectorSam3
==================

Uses Meta's SAM 3 to find objects by describing them in words, and returns
labelled image crops. It needs no training to start with: each output of the
task is a kind of object, and SAM 3 is prompted with its name.

SAM 3 only runs on Linux with an NVIDIA GPU. On such a machine install it with
``uv sync --extra sam3``.

.. code-block:: yaml

    find_hourglasses:
      type: collectra.ObjectDetectorSam3
      input: image
      output: [hourglass]
      threshold: 0.5

The prompt for an output is its name, with underscores read as spaces (so
``printed_label`` is prompted as "printed label"). Give a different prompt for
all the outputs with ``prompt``, or one for each with ``prompts``:

.. code-block:: yaml

    find_labels:
      type: collectra.ObjectDetectorSam3
      input: image
      output: [stamp, label]
      prompts:
        stamp: a round ink stamp
        label: a handwritten specimen label

Settings
--------

``model``
    A fine-tuned SAM 3 checkpoint. Leave it out to use Meta's released weights.
``version``
    Which of Meta's released weights to download: ``sam3.1`` (the default) or
    ``sam3``.
``threshold``
    Minimum confidence for a detection, from 0 to 1 (default 0.5).
``device``
    Where to run the model (default ``cuda``).

``resolution`` has no effect: SAM 3 always works at its own image size.

Fine-tuning
-----------

* ``run INPUTS...``: run the detector with the shared run options.
* ``train INPUTS...``: fine-tune on annotated result directories, using boxes
  only. The classes are the task's outputs, in the order the pipeline lists
  them. Pictures without any object are used as negative examples.

.. code-block:: bash

    collectra --pipeline pipeline.yaml task find_hourglasses run images --output results
    collectra --pipeline pipeline.yaml task find_hourglasses train training-data --validation validation --epochs 20

SAM 3 learns each class from its prompt, so every output needs a different
prompt when training. With a validation partition the best model is the one with
the highest box average precision on it; without one it is the one with the
lowest training loss. The trained weights are saved into the pipeline and set as
the task's ``model``. Training needs Meta's SAM 3 training sources, as
:doc:`image_segmentation` describes.
