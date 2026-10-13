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

File-backed text
----------------

Use ``path`` instead of ``data`` to store text in a separate UTF-8 file:

.. code-block:: yaml

    transcription:
      type: collectra.Text
      id: transcription-001
      path: texts/transcription.txt

Relative paths are resolved from the results directory. The file must exist.
Providing both ``data`` and ``path`` raises an error. Calling the Text object
returns the loaded contents; serialization preserves ``path`` and omits ``data``.
The GUI writes edits to the referenced file. Legacy file references in ``data``
remain supported, but new records should use ``path``.
