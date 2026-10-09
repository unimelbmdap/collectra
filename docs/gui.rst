GUI
===

The desktop GUI displays existing Collectra results using the selected
pipeline's task and artefact graph. It requires pywebview, a supported native
webview backend, and a desktop session.

Backend licences differ: PyQt is GPLv3 or commercially licensed, even though
pywebview itself is BSD-licensed. See :doc:`licensing` before choosing a backend
for a distributed application.

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

macOS Applications launcher
---------------------------

Create a clickable app for a pipeline with the Collectra icon:

.. code-block:: bash

    collectra --pipeline pipeline.yaml install-app --name "My Pipeline"

This creates ``~/Applications/My Pipeline.app``. Double-click it in Finder to
open the GUI without a Terminal window. You can also drag it to the Dock.
The launcher uses the current Python environment and the pipeline's absolute
path, and starts with the pipeline directory as its working directory. Keep
that environment and pipeline at their installed locations.

To open particular results on launch, pass their paths. To choose a destination
or a custom icon, use ``--app-dir`` or ``--icon``:

.. code-block:: bash

    collectra --pipeline pipeline.yaml install-app results --name "My Pipeline"
    collectra --pipeline pipeline.yaml install-app --name "My Pipeline" --app-dir /Applications --icon custom.icns

By default, the launcher uses ``icon`` from ``collectra_pipeline_metadata``
when configured, resolving relative paths from the pipeline directory:

.. code-block:: yaml

    collectra_pipeline_metadata:
      name: My Pipeline
      ext: collectra
      version: "1.0"
      icon: assets/pipeline.png

``--icon`` overrides this setting. If ``icon`` is absent, the pipeline's
``logo`` is used. Without either metadata setting, the Collectra icon is used.
Icons may be ``.icns`` or PNG files; PNGs are converted to macOS icons. The command requires macOS and
refuses to overwrite an existing app unless ``--force`` is supplied.
For example, reinstall with ``install-app --name "My Pipeline" --force``.
The replacement is built before the existing app is replaced. Writing to ``/Applications`` requires write access;
the default ``~/Applications`` is the user's Applications folder.

Sidebar and folder actions
---------------------------

When multiple results are open, the sidebar lists each result folder. Use the
search box above the list to filter folders by name.

Right-click a folder to reveal its ``results.yaml`` in the OS file manager
(Finder on macOS, Explorer on Windows), selected within its result folder.

Source control
--------------

Click the Git icon in the top-right toolbar to open source control. It shows
status and the repository path for the currently open results folder. Git finds
the repository in that folder or an ancestor; linked worktrees are supported.
Open a results folder before using these actions.

* **Status** shows ``git status``.
* **Pull** runs ``git pull --no-edit`` using the configured remote and branch.
* **Add** runs ``git add .`` from the repository root, staging all changes there.
* **Commit** opens a message popup and commits staged changes with that message.
* **Push** runs ``git push`` using the configured remote and branch.

Command output and errors appear in the viewer. Actions run one at a time.
Git must be installed, and remote authentication must already be configured;
terminal credential prompts are disabled. After pulling changes, reopen the
results folder to load the updated files into the viewer.

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
