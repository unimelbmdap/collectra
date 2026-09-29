Text Processing
===============

ConcatenateText
---------------

``collectra.ConcatenateText`` joins all input ``Text`` artefacts into one Text.
Unlike tasks that run separately for each artefact, it consumes the input
collection together. ``separator`` defaults to a newline.

.. code-block:: yaml

    join_text:
      type: collectra.ConcatenateText
      input: line_text
      output: transcription
      separator: "\n"

Commands
--------

``run`` is the only task command, with the :doc:`shared run options <index>`.
There is no ``train`` command.

.. code-block:: bash

    collectra --pipeline pipeline.yaml task join_text run results/page001.collectra
    collectra --pipeline pipeline.yaml task join_text run --help

The result must already contain ``line_text`` artefacts. See :doc:`ocr` for a
task that produces them and :doc:`../artefacts/text` for evaluation commands.
