LLM
===

``collectra.LLM`` sends image and/or text artefacts to a language model and
returns ``Text``. It is built on `pydantic-ai <https://ai.pydantic.dev>`_, which
comes with ``collectra[ml]``.

.. code-block:: yaml

    summarize:
      type: collectra.LLM
      model: YOUR_MODEL_IDENTIFIER
      input: transcription
      output: summary
      template: "Summarize this transcription: {transcription}"
      temperature: 0.2

Replace ``YOUR_MODEL_IDENTIFIER`` before using this example. Template
placeholders refer to input artefact names. ``prompt`` is accepted in place of
``template``, as in other tasks; set only one of them. For image inputs, the selected
model must support images. ``system``, ``preamble``, and ``max_tokens`` can also
be configured in YAML.

Choosing a model
----------------

Give the model as ``provider:model``. The provider's API key is read from the
environment, such as ``ANTHROPIC_API_KEY`` or ``OPENAI_API_KEY``; store it once
with the pipeline's ``env set`` command (see :doc:`../cli`).

=====================================  ==========================================
``anthropic:claude-sonnet-5-5``        Anthropic (``ANTHROPIC_API_KEY``)
``openai:gpt-5``                       OpenAI (``OPENAI_API_KEY``)
``google:gemini-2.5-pro``              Google (``GOOGLE_API_KEY``)
``mistral:pixtral-large-latest``       Mistral (``MISTRAL_API_KEY``)
``ollama:qwen2.5vl:7b``                A local Ollama server (``OLLAMA_BASE_URL``,
                                       e.g. ``http://localhost:11434/v1``)
``vllm:Qwen/Qwen2.5-VL-7B-Instruct``   A vLLM server (``VLLM_BASE_URL``)
=====================================  ==========================================

Any other provider pydantic-ai supports works the same way. A bare name such as
``claude-sonnet-5-5``, ``gpt-5``, ``gemini-2.5-pro``, ``mistral-large-latest``
or ``grok-4`` still works: the provider is added for you. Any other bare name is
an error that asks for the provider.

Local models run on your own machine or on a cluster GPU node, so no data
leaves it. For images, choose a vision model (for example Qwen2.5-VL or
Gemma 3). For several outputs or tools, the model and server must support tool
calling.

Prompt caching
--------------

With ``cache: true`` (the default), the parts of each request that repeat are
cached by providers that support it, such as Anthropic: the system prompt, the
tool definitions and, within one label, the conversation so far. The fixed text
of the template before the first image is cached as well, so it is reused from
page to page while a batch runs. Caching does not change the answers. Set
``cache: false`` to turn it off.

Several outputs
---------------

List more than one ``output`` node and the model answers field by field, one
string per node, named exactly as the node. Every node gets a ``Text``; a field
the model did not find is an empty string. Describe each field in the prompt
template. The model must support tool calling, because that is how the
structured answer is requested.

.. code-block:: yaml

    label_reader:
      type: collectra.LLM
      model: YOUR_MODEL_IDENTIFIER
      input: label_text
      output: [genus_draft, age_draft]
      template: |
        Read the genus and the geological age from this label.
        {label_text}

Looking things up
-----------------

An ``EntityLookup`` is a named list of known values that the model can search
before it answers. It is declared at the top level, not as a graph node, and a
task names it under ``tools``:

.. code-block:: yaml

    genera_lookup:
      type: collectra.EntityLookup
      file: genera.txt              # one entity per line, relative to the pipeline
      description: Genus names of graptolites; not exhaustive

    genus_reader:
      type: collectra.LLM
      model: YOUR_MODEL_IDENTIFIER
      input: [label_text, genus_image]
      output: genus
      tools: [genera_lookup]
      template: |
        Field: genus
        {label_text}
        {genus_image}

The model calls the lookup with the text it read, or a variant spelling, and
gets back up to ``limit`` candidates (default 10), best first, each with a
relative score; an exact match is marked. It can look up more than once, up to
``max_tool_calls`` times per prompt (default 5). Matching is BM25 over words and
character n-grams, so a misspelling still finds the entry and a single word finds
a longer entry that contains it. The index is built in memory; there are no index
files. A name in ``tools`` that no resource declares is reported when the
pipeline loads. Tools need a model that supports tool calling. Tools can be
combined with several outputs: the model searches as it needs to, then gives its
answer by calling one last tool whose fields are the output nodes, so one call
can read a whole label and check each field against its own list. With
``--batch-size``, prompts run their lookups side by side.

Commands
--------

``run`` is the only task command. There is no training or chat command.
It accepts the :doc:`shared run options <index>`.

At the end of every ``run`` that called a model, a table shows each LLM task's
model, calls, tokens and estimated cost, with a total:

.. code-block:: text

                                       LLM usage
                                                      Cache   Cache
    Task         Model            Calls      Input     read   write  Output     Cost
    ────────────────────────────────────────────────────────────────────────────────
    page_reader  claude-opus-5-5      1      3,000        0       0     900  $0.0300
    line_reader  claude-opus-5-5     21  1,234,567  800,000  90,000  34,567    $3.41

    Total                            22  1,237,567  800,000  90,000  35,467    $3.44

Costs are in US dollars, worked out from published prices by the
`genai-prices <https://github.com/pydantic/genai-prices>`_ package that
pydantic-ai uses. They are estimates: check your provider's billing for the
real amount. A model with no known price, such as a local one, shows
``unknown``, and a total that leaves such calls out starts with ``≥``.

Add ``--usage`` to also write each file's usage to ``usage.yaml`` in its
results folder: ``input_tokens`` (which includes cached tokens),
``output_tokens``, ``total_tokens``, ``cache_read_tokens``,
``cache_write_tokens``, ``requests`` and ``cost_usd``. ``cost_usd`` is left out
when the model's price is not known. Each run adds to the numbers already
there.

.. code-block:: bash

    collectra --pipeline pipeline.yaml task summarize run results/page001.collectra --usage
    collectra --pipeline pipeline.yaml task summarize run --help

The input ``transcription`` must already exist. To generate it and the summary
in one pass, run the full pipeline. See :doc:`canonicalisation` for tasks that
combine retrieval with an LLM.
