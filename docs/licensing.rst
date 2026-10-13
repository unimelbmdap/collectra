Licensing and optional backends
==================================

Collectra's own code is licensed under Apache-2.0. Dependencies, model weights,
remote services, and input collections retain their own licences and terms.
The project's licence does not replace those terms.

Installation choices
-----------------------

``pip install collectra`` installs the core without Ultralytics or drawyolo.
``pip install "collectra[yolo]"`` adds both for YOLO tasks and previews.
The lockfile records optional packages too; their presence there does not mean
that a default installation installs them.

.. list-table:: Backend licensing considerations
   :header-rows: 1
   :widths: 25 30 45

   * - Component
     - Code licence
     - What to check
   * - Ultralytics YOLO
     - AGPL-3.0, or separately negotiated commercial terms
     - The optional YOLO integration combines with Ultralytics. Installing an
       extra does not exempt that combination from applicable AGPL obligations.
   * - drawyolo
     - Apache-2.0
     - Also depends on Ultralytics, so it belongs in the YOLO extra.
   * - RF-DETR
     - Apache-2.0 for the open-source package
     - Select Apache-designated models; Plus components have separate terms.
   * - Torchvision / Transformers
     - BSD / Apache-2.0
     - Review the licence of each downloaded checkpoint separately.
   * - Surya
     - Version-dependent; 0.17.0 declares GPL-3.0-or-later code
     - Not installed by default. Its model weights have separate terms. Newer
       upstream code advertises Apache-2.0 but also changes the API; it is not
       automatically a drop-in replacement for Collectra's integration.
   * - pywebview
     - BSD
     - The selected GUI backend has additional terms. PyQt is GPLv3 or
       commercial; it is not LGPL. Native backends and PySide have different
       terms that must be checked for the target platform and bundled modules.

See the upstream sources for
`Ultralytics <https://www.ultralytics.com/license>`_,
`RF-DETR <https://github.com/roboflow/rf-detr#license>`_,
`Surya 0.17.0 <https://github.com/datalab-to/surya/blob/v0.17.0/README.md#commercial-usage>`_,
`current Surya <https://github.com/datalab-to/surya#commercial-usage>`_,
`PyQt <https://www.riverbankcomputing.com/software/pyqt/intro>`_, and
`pywebview backend options <https://pywebview.flowrl.com/guide/installation.html>`_.
Licences can change between versions; check the versions being distributed.

Distribution and contributions
----------------------------------

Preserve applicable dependency licences, copyright notices, and NOTICE files
when redistributing third-party software. A bundled desktop application or
container needs a review of its actual contents, including native libraries
and model files. The backend table is guidance, not a complete third-party
notice inventory or legal clearance of a release.

The default installation still contains dependencies with their own obligations,
including file-level copyleft licences such as MPL. Excluding YOLO does not
mean every installed file is Apache-2.0, or settle the legal status of the
integration code distributed with Collectra.

Before a release, maintainers should confirm the provenance of copied or adapted
code and the rights of contributors and their employers. Changes to licensing
or contribution terms require the relevant copyright holders' authority.
Any proposed contributor agreement should be reviewed by the institution.

For pipelines intended for redistribution, record the model's source, version,
licence and any required attribution alongside the pipeline. Include the terms
for fine-tuned weights where applicable. Software licensing does not replace
permissions needed for collection images, annotations, or hosted model services.

Bundled GUI assets
------------------

The GUI ships local copies of Annotorious, Cytoscape, Dagre, cytoscape-dagre,
Bootswatch/Bootstrap, EasyMDE, Font Awesome, and Source Sans Pro, in addition to
OpenSeadragon. Version and source information is recorded in
``collectra/gui/vendor/manifest.json``. Each dependency's licence is included
beside its assets. Bootswatch's remote font import is replaced with local font
files, and EasyMDE's automatic Font Awesome download is disabled.
