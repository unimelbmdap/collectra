Entity Resolution
=================

These tasks consume ``Text`` and produce ``Text`` containing a resolved name or
identifier. Both find candidates in their reference data with BM25 search,
then ask a language model to choose among them, so they need the model's
provider credentials.

BM25 is run locally and costs nothing. It matches whole words and also each
word's three-letter pieces (trigrams), so a misspelt or misread name such as
``Monograptas`` still finds ``Monograptus``. Rarer words and pieces count for
more.

LLMCanonicaliser
----------------

Matches text against an entity file using BM25 retrieval and an LLM.
Configure ``model``, ``entities`` (a text file containing one entity per line),
and a ``template`` using the input name and ``{entities}`` for retrieved candidates.
``count`` sets how many of the best-matching entities are offered (default 5).
The file is indexed the first time the task needs it. ``entities`` may instead
be a comma-separated list, which is offered whole.

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
``min_score``, and ``score_gap_eps``. The task builds or loads retrieval
indexes, and returns empty text if it finds no accepted match.

The model is only asked when the best candidate scores at least ``min_score``
(default 0: any candidate that shares a word or trigram with the query) and, if
``score_gap_eps`` is set, leads the second-best by at least that much. BM25
scores are higher for better matches and depend on the reference data, so look
at a few before choosing either value. ``score_floor``, the old maximum
embedding distance, is no longer accepted.

Commands: ``run`` only, with the shared run options::

    collectra --pipeline pipeline.yaml task resolve_irn run results/page001.collectra --usage
    collectra --pipeline pipeline.yaml task resolve_irn run --help

Here ``resolve_irn`` is the configured task name. There is no standalone CLI
command for training or rebuilding indexes.
