================================
Command Line Interface Reference
================================

Opening results in the GUI
--------------------------

Pass optional positional paths to populate the item list at startup::

    collectra --pipeline pipeline.yaml gui first.collectra second.collectra/results.yaml
    palynomorph gui predictions/*.palynomorph

Inputs can be result folders, files inside a result folder (such as
``results.yaml``), or parent directories containing result folders with the
pipeline's extension. Items retain argument order; parent-directory children
are sorted, and duplicate folders appear only once. The first item opens
automatically. Inputs must contain existing results; opening the GUI does not
create results for raw images.

Without paths, ``gui`` opens with the folder picker as before. ``--debug`` can
be used with or without input paths. The Python equivalent is::

    pipeline.launch_gui("first.collectra", "second.collectra", debug=True)

.. click:: collectra.main:app
   :prog: collectra
   :nested: full
