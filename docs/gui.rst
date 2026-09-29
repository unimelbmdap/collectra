GUI
===

The desktop GUI displays existing Collectra results using the selected
pipeline's task and artefact graph. It requires pywebview, a supported native
webview backend, and a desktop session.

Visual pipeline construction, training models from the GUI, and interactive
clustering are planned features. See :doc:`design_principles` for the broader
design goals and development direction.

Open results
------------

.. code-block:: bash

    collectra --pipeline pipeline.yaml gui
    collectra --pipeline pipeline.yaml gui results/page001.collectra
    collectra --pipeline pipeline.yaml gui results/page001.collectra/results.yaml
    collectra --pipeline pipeline.yaml gui results
    collectra --pipeline pipeline.yaml gui --help

Without paths, select a results folder in the GUI. You can supply multiple
result paths or parent directories. Result directories must use the extension
configured by the pipeline and contain ``results.yaml``. The GUI opens existing
results; it does not run a pipeline over raw images when opening a path.

An installed pipeline command offers the same interface::

    my-pipeline gui results

Inspect and edit artefacts
--------------------------

Select an item and a node in its graph to inspect the corresponding artefacts.
Images and crops display their source imagery and child boxes. Text artefacts
use the relevant text editor, and Links display their referenced targets.
Image views support box creation and editing. Edits are saved back to the
results.

For multichannel TIFFs, the viewer uses a preview of the channels. When the
channel count is divisible by three, view controls switch between RGB triplets;
the preview does not rewrite the source image.

Running and training tasks are documented under :doc:`tasks/index`.
Extraction and evaluation commands are documented under :doc:`artefacts/index`.

Debugging and Python usage
--------------------------

Add ``--debug`` to enable the webview's developer tools:

.. code-block:: bash

    collectra --pipeline pipeline.yaml gui results --debug

The Python equivalent is:

.. code-block:: python

    from collectra import Collectra

    pipeline = Collectra.from_file("pipeline.yaml")
    pipeline.launch_gui("results/page001.collectra", debug=True)

Custom displays
---------------

.. toctree::
   :maxdepth: 1

   artefact_display
