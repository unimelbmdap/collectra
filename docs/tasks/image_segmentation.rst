Image segmentation
==================

``ImageSegmentationSam3`` accepts ``Image`` and its subclasses and returns one
``ImageSegmentation`` artefact for every predicted instance. Each result retains
its input image as a parent and references a source-sized, binary 0/1 PNG mask.
Crop and orientation transforms are inverted before writing masks, so masks
remain aligned with the original source pixels. Segmentation inputs constrain
predictions to their existing mask.

.. code-block:: yaml

    collectra_pipeline_metadata:
      name: Segmentation
      ext: collectra
      version: "1.0"
    segment:
      type: collectra.ImageSegmentationSam3
      input: page
      output: region
      prompt: printed text region
      version: sam3.1
      threshold: 0.5
      device: cuda

SAM is promptable: a text concept identifies which instances to find. Without
``prompt``, output labels with underscores replaced by spaces become prompts.
For multiple concepts, supply a mapping of output labels to text prompts:

.. code-block:: yaml

    output: [heading, paragraph]
    prompts:
      heading: document heading
      paragraph: text paragraph

``model`` optionally names a local checkpoint relative to the pipeline directory.
Without it, the requested version's official checkpoint is downloaded on first
execution. The default is ``sam3.1``. This uses the image detector/segmenter
weights from the SAM3.1 checkpoint; it does not use video tracking or Object
Multiplex. ``sam3`` selects the older checkpoint.

Installation
------------

Use Python 3.12 for SAM3 training. SAM3 is installed separately from Collectra's
``ml`` extra. The GUI and dataset preparation do not require SAM3 or Torch.
Training requires a CUDA GPU; Meta lists PyTorch 2.7+ and CUDA 12.6+ as
prerequisites. Check the `upstream requirements
<https://github.com/facebookresearch/sam3#installation>`_ for compatible GPU and
PyTorch versions on your system.

The instructions below target Linux/HPC training. You need Git, `uv
<https://docs.astral.sh/uv/getting-started/installation/>`_, internet access for
installation and checkpoint downloads, and permission to run on a GPU node.
On a cluster, install/download on a node with network access and run training
inside a GPU allocation according to the cluster's instructions.

Create the environment
~~~~~~~~~~~~~~~~~~~~~~

Replace ``/path/to/workspace`` with a persistent directory you can write to.
This block creates a fresh environment; it stops on the first failure. Existing
source checkouts are retained. Use the latest Collectra code, which includes the
SAM3.1 checkpoint and training compatibility fixes.

.. code-block:: bash

    (
    set -e

    PROJECT_DIR=/path/to/workspace
    COLLECTRA_DIR="$PROJECT_DIR/collectra"
    SAM3_DIR="$PROJECT_DIR/sam3-source"

    mkdir -p "$PROJECT_DIR"
    if [ ! -d "$COLLECTRA_DIR" ]; then
        git clone https://github.com/unimelbmdap/collectra.git "$COLLECTRA_DIR"
    fi
    if [ ! -d "$SAM3_DIR" ]; then
        git clone https://github.com/facebookresearch/sam3.git "$SAM3_DIR"
    fi

    cd "$COLLECTRA_DIR"
    uv python install 3.12
    uv venv --python 3.12 .venv

    uv pip install --python "$COLLECTRA_DIR/.venv/bin/python" \
        -e "$COLLECTRA_DIR[ml]"
    uv pip install --python "$COLLECTRA_DIR/.venv/bin/python" \
        -e "$SAM3_DIR[train]" "setuptools<82" einops

    "$COLLECTRA_DIR/.venv/bin/python" - <<'PY'
    import sys
    from pathlib import Path
    import collectra
    import sam3
    import torch

    template = Path(sam3.__file__).parent / (
        "train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml"
    )
    assert sys.version_info[:2] == (3, 12), sys.version
    assert template.is_file(), f"Missing training template: {template}"
    print("Python:", sys.executable)
    print("Collectra:", collectra.__file__)
    print("SAM3:", sam3.__file__)
    print("Training template:", template)
    print("PyTorch:", torch.__version__)
    print("PyTorch CUDA build:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    PY
    )

Keep both source directories in place: these are editable installations.
SAM3 must be installed from a checkout so its training YAML files are available.
``setuptools<82`` provides the older ``pkg_resources`` API that SAM3 imports;
``einops`` supplies another required import.

Always pass ``--python`` when installing with ``uv pip``. An activated prompt
does not guarantee that a bare ``pip`` command installs into the same Python
used by Collectra. Using ``uv python install 3.12`` also avoids depending on
cluster Python module names or accidentally falling back to system Python when
a module fails to load. If downloads are blocked, ask the cluster administrator
for the supported Python 3.12 module and use its absolute interpreter path with
``uv venv --python /path/to/python3.12 .venv``.

If replacing an environment with the wrong Python version, deactivate it and
rename it before running the setup block, for example ``mv .venv .venv-old``
from the Collectra checkout. Choose an unused backup name. There is no need to
replace a working environment or rebuild prepared datasets.

Activate and check GPU access
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In each new shell, activate the environment (replace the path):

.. code-block:: bash

    source /path/to/workspace/collectra/.venv/bin/activate
    python --version
    command -v python
    command -v collectra

On a GPU compute node, run:

.. code-block:: bash

    nvidia-smi
    python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(), 'CUDA is unavailable'; print(torch.cuda.get_device_name(0))"

``CUDA available: False`` can be expected on a login node without GPU access.
On a GPU node, check your GPU allocation, NVIDIA driver and installed PyTorch
CUDA build. If a different PyTorch build is needed, use the `PyTorch installer
<https://pytorch.org/get-started/locally/>`_ and your cluster's guidance, targeting
the same ``.venv/bin/python``. A successful SAM3 import alone does not verify
GPU access or training.

Checkpoint access
~~~~~~~~~~~~~~~~~

Request access to the checkpoint repository for the version used by your task:
`SAM3.1 <https://huggingface.co/facebook/sam3.1>`_ (the default) or
`SAM3 <https://huggingface.co/facebook/sam3>`_. After approval, authenticate in
the activated environment using a Hugging Face token with read access:

.. code-block:: bash

    hf auth login

Download the default SAM3.1 checkpoint before starting an offline GPU job:

.. code-block:: bash

    python - <<'PY'
    from sam3.model_builder import download_ckpt_from_hf
    print(download_ckpt_from_hf(version="sam3.1"))
    PY

Use ``version="sam3"`` for an older SAM3 task. Keep the Hugging Face cache
accessible to the training job, or copy the downloaded checkpoint to persistent
storage and set the task's ``model`` to that file. Relative model paths are
resolved from the pipeline directory. Authorization errors require repository
access and authentication; reinstalling SAM3 will not resolve them.

Run your pipeline
~~~~~~~~~~~~~~~~~

With the environment activated, inspect the task command in your own pipeline:

.. code-block:: bash

    collectra --pipeline /path/to/pipeline.yaml task segment train --help

Replace ``segment`` with your task name, such as ``region_segmentor``. The
training examples below use ``segment`` from the example YAML above. You can
also install a shorter pipeline command:

.. code-block:: bash

    collectra --pipeline /path/to/pipeline.yaml install htr
    export PATH="$HOME/.local/bin:$PATH"
    htr task segment train --help

The launcher captures this environment's Python and the pipeline path; it does
not need environment activation afterward. Reinstall the launcher if either
path changes. Its name (``htr`` here) is independent of the task name.

Mask-supervised training
------------------------

Yes: Meta's image training loader accepts instance masks. Each ground-truth
instance should be an ``ImageSegmentation`` in the task's output node, with
``parents`` referencing its input image and ``data`` referencing the same source
image. Masks follow Collectra's source-sized 0/1 PNG convention. The output node
name supplies the category, and the configured prompt supplies its text concept.
Annotate all instances of each concept in each training image: SAM's COCO loader
treats these as exhaustive annotations.

``train`` automatically prepares its dataset under ``OUTPUT/dataset`` before
launching training; no separate preparation command is needed.
The export contains ``train`` and ``valid`` directories, RGB PNG images, and
``_annotations.coco.json`` files with instance masks encoded as lossless COCO
RLE. Holes and disconnected regions are retained. The ``partition`` values
select validation examples; ``--exclude NAME`` omits one partition from both
training and validation, matching the other training tasks. All other examples
are training data. At least one labelled training image is required.

To prepare data and launch Meta's image fine-tuning trainer:

.. code-block:: bash

    collectra --pipeline /path/to/pipeline.yaml task segment train \
        /path/to/results \
        --output /path/to/sam-training \
        --validation validation \
        --epochs 20 --batch 1 --num-gpus 1

For example, reserve validation images and leave test images unused:

.. code-block:: bash

    htr task region_segmentor train /path/to/results \
        --output /path/to/sam-training \
        --validation validation \
        --exclude test

Partition changes take effect when exporting data. Omit ``--reuse-data`` once
to apply new exclusions to an existing training directory.

Use ``--max-items 100`` for a smaller run. This caps the **combined total** of
training and validation images after exclusions, retaining every annotated
instance in each selected image. Images are selected in discovery/input order,
and source loading stops once the cap is reached. The cap does not reserve a
validation quota, so a small prefix may contain no validation images. Omit the
option to use all images. Values must be positive when preparing data.
When an export exists, ``--reuse-data`` takes precedence and ignores
``--max-items``, leaving the existing export unchanged. To change the cap, omit
``--reuse-data`` once. When the export is missing, preparation uses the supplied
cap and partition options normally.

``--output`` is optional, as with YOLO and RT-DETR. If omitted, training uses a
timestamped directory in the current working directory. Supply a custom
directory with ``--output PATH``; positional paths are training inputs.
Logs and prepared data are retained by default. ``--no-keep-log`` removes the
training directory after the trained checkpoint has been saved to the pipeline.
Preparation prints result-file counts, the current file, image/mask export
counts, split totals and elapsed times. It runs on the CPU before GPU training
starts. Checkpoint resolution, configuration output and trainer startup are
reported separately. Progress is flushed immediately and the trainer runs with
unbuffered output so these messages also appear in redirected cluster logs.

To use an already prepared dataset directly, without scanning source results,
checking for changes or rebuilding images and masks::

    collectra --pipeline /path/to/pipeline.yaml task segment train \
        --output /path/to/sam-training --reuse-data

``--reuse-data`` requires ``--output``. If both ``dataset/train`` and
``dataset/valid`` contain ``_annotations.coco.json``, it uses that export directly
without source/freshness checks. Input paths can then be omitted; supplied input
paths, partition settings and ``--max-items`` are ignored. If either annotation
file is missing, it prints a message and prepares the dataset from the supplied
inputs instead, applying the partition options and cap. With no export and no
inputs, it reports that input paths are required. Normal training without this
flag always prepares the dataset.

The adapter uses the installed upstream Roboflow image-training template,
enables segmentation in the model, dataset and collator, and adds mask focal and
Dice losses. Generated configuration is saved as ``sam3-training.yaml`` in the
output directory. Without validation images, the trainer uses ``train_only``;
training images are not reused as validation. After training, the best
checkpoint is copied into the pipeline directory and the task's ``model`` is
updated in ``pipeline.yaml``. The task can load both official full checkpoints
and fine-tuned image-only checkpoints.

Metrics, W&B and model selection
----------------------------------

Add ``--wandb`` to send epoch metrics to Weights & Biases. Without this flag,
training does not initialize W&B; local CSV reporting is always enabled.
Authenticate with ``wandb login`` beforehand, or configure your site's
``WANDB_API_KEY``. For example:

.. code-block:: bash

    export WANDB_ENTITY=my-team
    htr task region_segmentor train /path/to/results \
        --output /path/to/sam-training \
        --validation validation --exclude test --wandb

As with the other training tasks, the W&B project is
``<pipeline-directory-name>-<task-name>`` (for example,
``HTR.collectra-region_segmentor``). The run name is the output directory name.
Entity selection is left to W&B's configured account or ``WANDB_ENTITY``; SAM3
does not override it. The explicit project takes precedence over ``WANDB_PROJECT``.
On a cluster without network access, set ``WANDB_MODE=offline`` before training
and sync the W&B run directory afterward.

``OUTPUT/metrics.csv`` contains one row per epoch. Training and validation
metrics merge into the same row; validation columns are empty for epochs without
validation. W&B receives the same metrics, using the one-based ``epoch`` as its
chart axis. Native trainer metrics are included alongside ``train/loss`` and
``val/mask_iou``.

``val/mask_iou`` measures foreground-mask intersection over union. For each
image/concept query, predictions are filtered using the task's ``threshold``
(including SAM's presence score), mask logits are resized to the validation mask
size and thresholded at zero, and instance masks are combined into foreground.
Intersections and unions are summed across validation queries and GPUs before
division. False foreground on empty images is penalized. Queries with missing
ground-truth masks are omitted. An entirely empty foreground prediction and
target score 1; no evaluated queries is an error for best-model selection.
This measures pixel coverage, not instance matching or instance-segmentation AP.

Validation runs after every epoch, including the first and last. When validation
IoU strictly improves, ``OUTPUT/training/checkpoints/best.pt`` is replaced
atomically with the model's raw state dictionary. ``best.json`` records its epoch,
selection metric and value. No optimizer, scheduler or scaler state is saved,
and native periodic/last-epoch checkpoints are disabled. After training, Collectra
copies these best weights into the pipeline and updates the task's ``model``.

Without validation images, selection falls back to the lowest training loss and
is recorded as ``train/loss`` in ``best.json``. These weights support inference
and further fine-tuning, but do not resume optimizer state. ``--reuse-data``
continues to mean reuse prepared images/masks; an old optimizer checkpoint in
the output directory is not resumed. ``--no-keep-log`` removes the local CSV and
training directory after the chosen weights have been copied into the pipeline.

Upstream references: `training guide
<https://github.com/facebookresearch/sam3/blob/main/README_TRAIN.md>`_,
`COCO mask loader
<https://github.com/facebookresearch/sam3/blob/main/sam3/train/data/coco_json_loaders.py>`_,
and `image-training configuration
<https://github.com/facebookresearch/sam3/blob/main/sam3/train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml>`_.

Troubleshooting
---------------

Use the first traceback in the log to identify the cause. The final
``subprocess.CalledProcessError`` only reports that the trainer failed. Capture
both stdout and stderr when reporting a problem:

.. code-block:: bash

    (
    set -o pipefail
    collectra --pipeline /path/to/pipeline.yaml task segment train \
        --output /path/to/sam-training --reuse-data 2>&1 | tee sam-training.log
    )

This example assumes the dataset was already exported successfully. Use the
initial training command with an input directory for a new dataset. On HPC,
prefer persistent scratch storage for training output; ``/tmp`` can be local to
a node and may disappear between jobs. Creating an empty output directory does
not create a reusable dataset. ``--reuse-data`` reuses data, not training state.

Common setup problems:

* ``No module named sam3``: compare the traceback's Python path with the
  installation target. Install into Collectra's venv using ``uv pip --python``.
* ``No module named pkg_resources``: install ``"setuptools<82"`` into that
  environment. A deprecation warning from ``pkg_resources`` is not a failure.
* ``No module named einops``: install ``einops`` into that environment.
* Missing ``roboflow_v100_full_ft_100_images.yaml``: install SAM3 with
  ``-e "/path/to/sam3-source[train]"`` and retain the source checkout.
* Missing/unexpected checkpoint keys, ``Expected grad to be disabled``, or
  DDP's ``Expected to have finished reduction``: first ensure your Collectra
  checkout includes the SAM3 training fixes. The adapter handles SAM3.1 image
  checkpoint conversion, replaces inference-only ViT MLP operations for
  training, and freezes the semantic head not supervised by instance-mask
  losses. Do not bypass these errors by disabling gradients or loading all
  checkpoint weights non-strictly. If the error persists on updated code,
  capture the full log rather than assuming it has the same cause.

For support, include the task YAML, full first traceback, verification output
above, ``nvidia-smi`` output, and the Collectra and SAM3 revisions (run
``git -C /path/to/checkout rev-parse HEAD`` for each). This distinguishes
installation, checkpoint, GPU and training-code problems without guessing from
the final wrapper error.
