LLM
===

``collectra.LLM`` sends image and/or text artefacts to a language model and
returns ``Text``. Configure a model identifier supported by your ``llmloader``
installation and the credentials required by its provider.

.. code-block:: yaml

    summarize:
      type: collectra.LLM
      model: YOUR_MODEL_IDENTIFIER
      input: transcription
      output: summary
      template: "Summarize this transcription: {transcription}"
      temperature: 0.2

Replace ``YOUR_MODEL_IDENTIFIER`` before using this example. Template
placeholders refer to input artefact names. For image inputs, the selected
model must support images. ``system``, ``preamble``, and ``max_tokens`` can also
be configured in YAML.

Commands
--------

``run`` is the only task command. There is no training or chat command.
It accepts the :doc:`shared run options <index>`; add ``--usage`` to record
token-usage information supplied by the backend.

.. code-block:: bash

    collectra --pipeline pipeline.yaml task summarize run results/page001.collectra --usage
    collectra --pipeline pipeline.yaml task summarize run --help

The input ``transcription`` must already exist. To generate it and the summary
in one pass, run the full pipeline. See :doc:`canonicalisation` for tasks that
combine retrieval with an LLM.
