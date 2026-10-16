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

Meta's SAM3 sources and checkpoints are installed separately. They are not
added to Collectra's default install or its general ``ml`` extra. Consult the
`upstream prerequisites and model access instructions
<https://github.com/facebookresearch/sam3#installation>`_. Meta currently lists
Python 3.12+, PyTorch 2.7+, and a CUDA GPU as prerequisites.

.. code-block:: bash

    pip install "sam3[train] @ git+https://github.com/facebookresearch/sam3.git"

Importing the task, opening its pipeline in the GUI, and preparing training data
do not import SAM or Torch. Inference and training require the model backend.

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
select validation examples; ``--exclude`` omits a partition. All other examples
are training data. At least one labelled training image is required.

To prepare data and launch Meta's image fine-tuning trainer:

.. code-block:: bash

    collectra --pipeline pipeline.yaml task segment train results --output sam-training --validation validation --epochs 20 --batch 1 --num-gpus 1

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

    htr task region_segmentor train --output /tmp/htr --reuse-data

``--reuse-data`` requires ``--output`` and uses its existing ``dataset/train``
and ``dataset/valid`` exports. Input paths can be omitted; any supplied input
paths and ``--validation``/``--exclude`` settings are ignored because the
existing dataset already defines the splits. The export must have completed
successfully. Normal training without this flag still prepares the dataset.

The adapter uses the installed upstream Roboflow image-training template,
enables segmentation in the model, dataset and collator, and adds mask focal and
Dice losses. Generated configuration is saved as ``sam3-training.yaml`` in the
output directory. Without validation images, the trainer uses ``train_only``;
training images are not reused as validation. After training, the final
checkpoint is copied into the pipeline directory and the task's ``model`` is
updated in ``pipeline.yaml``. The task can load both official full checkpoints
and fine-tuned image-only checkpoints.

Upstream references: `training guide
<https://github.com/facebookresearch/sam3/blob/main/README_TRAIN.md>`_,
`COCO mask loader
<https://github.com/facebookresearch/sam3/blob/main/sam3/train/data/coco_json_loaders.py>`_,
and `image-training configuration
<https://github.com/facebookresearch/sam3/blob/main/sam3/train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml>`_.
