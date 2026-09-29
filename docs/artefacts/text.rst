Text
====

``collectra.Text`` holds text produced by OCR, language models, or text-processing
tasks. For example:

.. code-block:: yaml

    transcription:
      type: collectra.Text
      id: transcription-001
      parents: page-001
      data: "Collected in Melbourne."

Commands
--------

``evaluate PREDICTED_FOLDER GOLD_FOLDER`` is the only Text command. It uses
``difflib.SequenceMatcher`` similarity, not character-error rate or word-error
rate. The :doc:`shared evaluation options <index>` control matching and CSV output.

.. code-block:: bash

    collectra --pipeline pipeline.yaml artefact transcription evaluate predictions gold --output text-metrics.csv
    collectra --pipeline pipeline.yaml artefact transcription evaluate --help

There is no Text ``extract`` CLI command. Text is stored in ``results.yaml``
and can be inspected in the :doc:`GUI <../gui>` or accessed in Python.
