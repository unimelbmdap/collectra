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
--------

``run`` is the only task command, with the :doc:`shared run options <index>`.
There is no ``train`` command.

.. code-block:: bash

    collectra --pipeline pipeline.yaml task read_lines run results/page001.collectra
    collectra --pipeline pipeline.yaml task read_lines run --help

The example expects ``lines`` crops in the results. To OCR whole images, use
the root image node as ``input`` instead. Use :doc:`text` to join OCR output and
:doc:`../artefacts/text` to evaluate it.
