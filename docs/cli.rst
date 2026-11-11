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

Partitioning results
--------------------

Installed pipeline commands expose ``partition`` directly::

    htr partition results --train 80% --validation 20% --seed 42
    htr partition results --train 0.8 --val 0.1 --test 0.1
    htr partition results --partition training=80% --partition heldout=20%

The equivalent generic command is::

    collectra --pipeline pipeline.yaml partition results --train 80% --validation 20%

Inputs may be individual result folders, ``results.yaml`` files, or parent
directories. Parent directories are searched recursively using the pipeline's
result extension. Duplicate results are handled once. Values may be percentages,
fractions or integer counts; use one format throughout, with a total of at most
100%, 1 or the number of results respectively. At least one partition is required.
``--val`` assigns the label ``val``; ``--validation`` assigns ``validation``.
Custom labels use repeated ``--partition NAME=VALUE`` options.

The command updates the ``partition`` of the ``results`` entry (the file's
header) in each selected result. Results left over after the specified allocations keep their existing
partition. Inputs are sorted before shuffling, so a fixed seed reproduces the
same assignment for the same set of results.

Environment variables
---------------------

A pipeline can keep environment variables, such as API keys, for the user who
runs it. They are stored outside the pipeline, in the user's config folder, in
a file named after the pipeline (``~/.config/collectra/env/hespi.yaml`` on
Linux, ``~/Library/Application Support/collectra/env/hespi.yaml`` on macOS),
which only that user can read. Every time the pipeline is loaded, its variables
are put into the environment::

    hespi env set ANTHROPIC_API_KEY             # asks for the value without showing it
    hespi env set HF_HOME /data/scratch/hf      # or give the value
    hespi env set ANTHROPIC_API_KEY --private   # keep it in the system keyring
    hespi env show                              # --reveal shows private values
    hespi env set HF_HOME /data/hf --override   # use it even when the shell sets one
    hespi env override HF_HOME --off            # switch override off (or on, without --off)
    hespi env delete HF_HOME
    hespi env clear                             # asks first; --yes does not

Typing a value at the prompt keeps it out of the shell's history. ``--private``
keeps the value in the operating system's keyring (the macOS Keychain, Windows
Credential Manager or the Linux Secret Service) and writes only its name to the
file; ``show`` masks it. Machines without a keyring service, such as most
cluster nodes, cannot store private values: ``set --private`` says so rather
than writing the value to the file.

A variable already set in the shell takes precedence over the stored one, so a
job script can still change it. A variable stored with ``--override`` takes
precedence over the shell instead; deleting it, or switching its override off,
brings the shell's value back. Loading the pipeline, ``env set`` and
``env show`` all say when the shell's value and the stored one differ. ``set``
restates a variable in full, so give ``--private`` and ``--override`` again when
changing a value that has them.

The generic form of these commands is
``collectra --pipeline pipeline.yaml env ...``; in Python, ``pipeline.env``
gives the same operations. The GUI shows them under :ref:`Settings <gui-settings>`.

Opening results in the GUI
--------------------------

Pass optional positional paths to populate the item list at startup::

    collectra --pipeline pipeline.yaml gui first.collectra second.collectra/results.yaml
    palynomorph gui predictions/*.palynomorph

Inputs can be result folders, files inside a result folder (such as
``results.yaml``), parent directories containing result folders with the
pipeline's extension, or raw input files and folders of them, such as images.
Items retain argument order; parent-directory children are sorted, and
duplicate folders appear only once. The first item opens automatically.
Opening a raw input creates a page for it, as described under
:ref:`gui-open-images`, and prints a one-line summary in the terminal. An
input the pipeline does not accept is skipped when another input opened a page;
if nothing opened, it is an error.

Without paths, ``gui`` opens with the folder picker as before. ``--debug`` can
be used with or without input paths. The Python equivalent is::

    pipeline.launch_gui("first.collectra", "second.collectra", debug=True)
