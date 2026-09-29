Entity Resolution
=================

These tasks consume ``Text`` and produce ``Text`` containing a resolved name or
identifier. Both use configured language and embedding models, so they require
their provider credentials and reference data.

LLMCanonicaliser
----------------

Matches text against an entity file using embedding retrieval and an LLM.
Configure ``model``, ``entities`` (a text file containing one entity per line),
and a ``template`` using the input name and ``{entities}`` for retrieved candidates.
``embedding_model`` and ``count`` control candidate retrieval.

Commands: ``run`` only, with the :doc:`shared run options <index>`::

    collectra --pipeline pipeline.yaml task canonicalise run results/page001.collectra --usage
    collectra --pipeline pipeline.yaml task canonicalise run --help

Here ``canonicalise`` is the configured name of a
``collectra.LLMCanonicaliser`` task. There is no ``train`` command.

IRNResolver
-----------

Resolves text to an IRN identifier from an Excel reference file. Configure
``type: collectra.IRNResolver``, ``model``, ``source`` (the workbook),
``card_template`` (reference-row fields), and ``query_template`` (input Text
names). Additional controls include ``irn_column``, ``index_filters``, ``count``,
``score_floor``, and ``score_gap_eps``. The task builds or loads retrieval
indexes, and returns empty text if it finds no accepted match.

Commands: ``run`` only, with the shared run options::

    collectra --pipeline pipeline.yaml task resolve_irn run results/page001.collectra --usage
    collectra --pipeline pipeline.yaml task resolve_irn run --help

Here ``resolve_irn`` is the configured task name. There is no standalone CLI
command for training or rebuilding indexes.
