Tasks
=====

Tasks transform artefacts. Choose a task class in ``pipeline.yaml``, give it a
name, and connect its ``input`` and ``output`` artefact nodes. Commands use the
configured name, not the class name:

.. code-block:: bash

    collectra --pipeline pipeline.yaml task --help
    collectra --pipeline pipeline.yaml task detect_lines --help
    collectra --pipeline pipeline.yaml task detect_lines run --help

.. toctree::
   :maxdepth: 2

   object_detection
   line_detection
   image_classification
   ocr
   llm
   canonicalisation
   image_orientation
   text

Command overview
----------------

.. list-table:: Built-in task commands
   :header-rows: 1
   :widths: 65 35

   * - Task class
     - Commands
   * - ObjectDetectionYOLO
     - ``run``, ``train``
   * - ObjectDetectionRFDETR
     - ``run``, ``train``
   * - ObjectDetectionDETR
     - ``run``, ``train``
   * - LineDetectorRLSA
     - ``run``
   * - LineDetectorSurya
     - ``run``
   * - ImageClassifierYOLO
     - ``run``, ``train``
   * - ImageClassifierTorchvision
     - ``run``, ``train``
   * - ImageClassifierHuggingFace
     - ``run``, ``train``
   * - OCRSuraya
     - ``run``
   * - LLM
     - ``run``
   * - LLMCanonicaliser
     - ``run``
   * - IRNResolver
     - ``run``
   * - ImageOrienter
     - ``run``
   * - ConcatenateText
     - ``run``

Running tasks
-------------

Every task supports ``run INPUTS...`` and ``run --help``. Shared options are
``--output``, ``--verbose``, ``--usage``, and ``--render``. ``--usage`` records
LLM token usage when supported, and ``--render`` requires Graphviz.
Task-level ``run`` executes only the selected task. Its input artefacts must be
present in the supplied results, or be the root image supplied as a raw file.

To execute a task and then its downstream tasks, use the pipeline command:

.. code-block:: bash

    collectra --pipeline pipeline.yaml run results/page001.collectra --task detect_lines

Training tasks
--------------

Only the detection and classification tasks listed above expose ``train``.
Training consumes annotated Collectra result directories. Options differ by
backend; use the particular task's ``train --help``. Torchvision and Hugging Face
classifiers support opt-in experiment logging with ``--wandb``. RF-DETR exposes
``--no-wandb`` to disable its default logging.

Training updates the task's model in the pipeline and retains training logs by
default. See :doc:`../quickstart` for preparing annotations and a complete
training example. Evaluation and image extraction are
:doc:`artefact commands <../artefacts/index>`, not task commands.
