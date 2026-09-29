Link
====

``collectra.Link`` refers to another artefact through its first parent ID.
Classifiers use it to label an image or crop without duplicating that image:

.. code-block:: yaml

    Pollen:
      type: collectra.Link
      id: classification-001
      parents: image-001

The referenced image must also be present in the results. Python's ``resolve()``
returns the target artefact. Links are used for classification training labels
as well as predictions.

Commands
--------

``evaluate PREDICTED_FOLDER GOLD_FOLDER`` is the only Link command. Use the
class node name and the :doc:`shared evaluation options <index>`:

.. code-block:: bash

    collectra --pipeline pipeline.yaml artefact Pollen evaluate predictions gold --output pollen-metrics.csv
    collectra --pipeline pipeline.yaml artefact Pollen evaluate --help

For unresolved Links, comparison uses the parent IDs, so predicted and reference
results should preserve source IDs. If both Links have resolved targets, the
comparison delegates to the target artefacts; those targets must support
evaluation. Whole Images do not implement an evaluation score.

Link nodes do not expose ``extract``. Extract images from their source Image or
ImageCrop node instead.
