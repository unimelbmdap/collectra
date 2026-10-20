OCR
===

OCRSuraya
---------

``collectra.OCRSuraya`` recognizes text in images or crops and returns ``Text``
artefacts. The class name is spelled ``OCRSuraya``; the underlying backend is
Surya. Install its ``surya-ocr`` package; weights load on first use.
See :doc:`../licensing` for version-dependent code and model-weight terms.

.. code-block:: yaml

    read_lines:
      type: collectra.OCRSuraya
      input: lines
      output: line_text

Commands
~~~~~~~~

``run`` is the only task command, with the :doc:`shared run options <index>`.
There is no ``train`` command.

.. code-block:: bash

    collectra --pipeline pipeline.yaml task read_lines run results/page001.collectra
    collectra --pipeline pipeline.yaml task read_lines run --help

The example expects ``lines`` crops in the results. To OCR whole images, use
the root image node as ``input`` instead. Use :doc:`text` to join OCR output and
:doc:`../artefacts/text` to evaluate it.

OCRHuggingFace
--------------

``collectra.OCRHuggingFace`` transcribes images or crops with any Transformers
model that loads natively with ``AutoModelForImageTextToText`` and returns one
``Text`` artefact per input. Remote model code is never run, so a Hub repository
must use a model type built into the installed ``transformers`` version.

.. list-table:: Example models
   :header-rows: 1

   * - Model
     - Suggested ``prompt``
   * - ``deepseek-community/DeepSeek-OCR-2`` (default)
     - ``Free OCR.`` (default)
   * - ``PaddlePaddle/PaddleOCR-VL-1.6``
     - ``OCR:``
   * - ``zai-org/GLM-OCR``
     - ``Text Recognition:``
   * - ``lightonai/LightOnOCR-2-1B``
     - ``""`` (image only)
   * - ``baidu/Qianfan-OCR``
     - ``OCR``

Use ``deepseek-community/DeepSeek-OCR-2`` rather than
``deepseek-ai/DeepSeek-OCR-2``, which requires remote code. Models with a chat
template receive the image and ``prompt`` as a user message; others receive
``prompt`` directly, with the image token prepended if it is absent.

.. code-block:: yaml

    line_reader:
      type: collectra.OCRHuggingFace
      input: line_image
      output: line_text
      model: PaddlePaddle/PaddleOCR-VL-1.6
      prompt: "OCR:"
      max_new_tokens: 256

Options are ``model`` (Hub ID or ``save_pretrained`` directory, relative to the
pipeline), ``prompt``, ``max_new_tokens``, ``device`` (automatic when omitted)
and ``dtype`` (``auto`` uses the checkpoint's precision, or e.g. ``bfloat16``).

Commands
~~~~~~~~

``run`` takes the :doc:`shared run options <index>`. ``train`` fine-tunes the
model on ``Text`` outputs paired with the images they were read from, using the
same prompt as inference so the loss covers only the transcription.

.. code-block:: bash

    collectra --pipeline pipeline.yaml task line_reader train data/ \
        --validation validation --exclude test --epochs 5 --batch 4
    collectra --pipeline pipeline.yaml task line_reader train --help

Training exports images and ``labels.jsonl`` files to ``OUTPUT/dataset`` (reuse
them with ``--reuse-data``), writes ``metrics.csv`` with the validation loss and
character error rate, and copies the model with the lowest character error rate
into the pipeline directory, updating ``model``. Without ``--validation`` the
last epoch is kept. For large models, ``--lora`` trains low-rank adapters
(requires ``peft``) and merges them into the saved model;
``--gradient-checkpointing``, ``--freeze-vision`` and ``--accumulate`` reduce
memory use further.
