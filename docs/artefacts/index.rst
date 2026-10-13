Artefact Types
==============

Artefacts are the data flowing through a pipeline: source images, crops, text,
and class labels. ``results.yaml`` stores their types, IDs, data, and parent
relationships. Tasks produce artefacts; artefact commands operate on saved
results, using the configured node name:

.. code-block:: bash

    collectra --pipeline pipeline.yaml artefact --help
    collectra --pipeline pipeline.yaml artefact lines --help

For example, ``lines`` names a node of ImageCrop artefacts, rather than an
individual crop ID or the class name ``ImageCrop``.

.. toctree::
   :maxdepth: 1

   image
   image_crop
   image_segmentation
   text
   link

Command overview
----------------

.. list-table:: Built-in artefact commands
   :header-rows: 1
   :widths: 20 25 55

   * - Type
     - Commands
     - Behaviour
   * - Image
     - ``extract``, ``evaluate``
     - Extraction is supported. The inherited evaluation command has no
       comparison implementation for whole Image artefacts.
   * - ImageCrop
     - ``extract``, ``evaluate``
     - Export crop images and evaluate box intersection-over-union.
   * - Text
     - ``evaluate``
     - Compare text using sequence similarity.
   * - Link
     - ``evaluate``
     - Evaluate references; see the Link page for comparison behaviour.

Evaluation
----------

``evaluate PREDICTED_FOLDER GOLD_FOLDER`` compares matching result directories
for the selected node. Shared options are:

* ``--threshold``: minimum matching score, default ``0.5``.
* ``--match-by-order``: use positional matching instead of the normal matching.
* ``--output``: write evaluation metrics to a CSV file.

Reports are also printed to the terminal. For extraction, pass individual result
directories or a shell-expanded glob; ``extract`` does not scan an arbitrary
parent directory for results. Every command supports ``--help``.

Artefacts do not expose ``run`` or ``train``; those are :doc:`task commands
<../tasks/index>`. See :doc:`../gui` to inspect and edit result artefacts visually.
