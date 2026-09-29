Design principles and development direction
===============================================

Collectra aims to make it easier for researchers and institutions to turn
collection images and documents into structured, searchable data. The design
grew from collection-specific workflows such as Hespi and Grapto: researchers
should be able to reuse their components and adapt them to new collections.

These principles reflect the vision presented at the ARDC Summer School.
**Collectra is still in development, and many of the features in that vision
are still to come.** The goals below describe the intended direction, rather
than a promise that every workflow is available now. There is no release date
implied for the planned features.

Modular, reusable pipelines
-------------------------------

Each task should perform a distinct operation, with artefacts connecting its
inputs and outputs to other tasks. Researchers can combine object detection,
classification, OCR, and LLM prompts to fit their collection, reuse an existing
pipeline, or implement a new task in Python. Multiple LLM tasks can turn an
image into a transcription and then into structured text, for example.

Today, pipelines are configured in YAML and run through the CLI or Python.
A visual editor for connecting and configuring tasks is planned. The intended
task range also includes landmark/pose detection, segmentation, and oriented
bounding boxes; these are not documented as available built-in tasks.
See :doc:`tasks/index` for the current tasks and their commands.

Result bundles that preserve context
----------------------------------------

Extracted data should retain its relationship to the source material. A crop,
its transcription, and a classification should remain connected so that a
researcher can inspect the evidence behind a result.

Collectra currently saves artefacts and their parent relationships in result
directories containing ``results.yaml`` and associated files. Predictions and
training annotations use the same result format. This allows reviewed results
to become training examples when they have the annotations required by the
chosen task.

RO-Crate-compatible packaging is a design goal. The current Collectra result
format should not be assumed to be a complete RO-Crate. See
:doc:`artefacts/index` for the current artefact types and commands.

Human review as part of the process
---------------------------------------

AI predictions need the judgement of researchers and curators. Reviewing and
correcting results is part of the intended workflow, before those results are
used in research, added to a database, or reused for training.

The current :doc:`gui` supports inspecting results and editing annotations,
including bounding boxes and text. The broader goal is a connected workflow
from prediction through verification to export into collection databases;
general database export is planned work.

An iterative path to better models
--------------------------------------

Researchers should be able to start with pretrained models, predict on their
own material, correct the results, and fine-tune on those reviewed examples.
Keeping predictions and annotations in the same format reduces the conversion
work between these stages. Held-out validation data is still needed to assess
whether retraining improves the model.

Supported detection and classification tasks currently expose CLI ``train``
commands. Training a selected task directly from the GUI is planned. See the
:doc:`quickstart` for preparing annotations and training through the CLI.

Exploration that guides annotation
--------------------------------------

Clustering should help researchers discover groups of similar items, such as
specimen labels with the same layout or recurring depictions in artwork.
Embeddings can also guide which examples to review first: the proposed
furthest-first traversal would prioritise diverse examples, helping annotators
cover variation in the collection with less repeated effort.

Interactive clustering, visual exploration of those groups, and diversity-based
annotation ordering are planned workflows. They are not currently exposed as
the GUI's proposed ``Cluster`` action. Existing internal similarity utilities
do not constitute this complete workflow.

Working with Collectra today
--------------------------------

Start with the :doc:`quickstart`, choose from the available :doc:`tasks/index`,
and use the :doc:`gui` to inspect saved results. The task and artefact command
pages describe current interfaces; the design goals above explain where the
project is heading.
