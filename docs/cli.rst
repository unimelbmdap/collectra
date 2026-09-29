================================
Command Line Interface Reference
================================

The CLI is built from the tasks and artefacts in the selected pipeline. Supply
``--pipeline`` with a YAML file or a directory containing ``pipeline.yaml``.
Use these commands to inspect the options available for your pipeline::

    collectra --pipeline pipeline.yaml --help
    collectra --pipeline pipeline.yaml run --help
    collectra --pipeline pipeline.yaml task --help
    collectra --pipeline pipeline.yaml artefact --help

Run a pipeline over images, or run one configured task::

    collectra --pipeline pipeline.yaml run images --output results
    collectra --pipeline pipeline.yaml task detect_lines run results/page001.collectra

Replace ``detect_lines`` with a task name from your YAML configuration. Task
commands expose their own help, including backend-specific training options::

    collectra --pipeline pipeline.yaml task classify train --help

See :doc:`quickstart` for complete pipeline and training examples.

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
