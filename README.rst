.. image:: https://curly-adventure-5j8lz2j.pages.github.io/_images/collectra-banner.png

.. start-badges

|testing badge| |coverage badge| |docs badge| |black badge| |torchapp badge|

.. |testing badge| image:: https://github.com/unimelbmdap/collectra/actions/workflows/testing.yml/badge.svg
    :target: https://github.com/unimelbmdap/collectra/actions

.. |docs badge| image:: https://github.com/unimelbmdap/collectra/actions/workflows/docs.yml/badge.svg
    :target: https://unimelbmdap.github.io/collectra
    
.. |black badge| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    
.. |coverage badge| image:: https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/unimelbmdap/f1d9191993105301fd6f813fe1e659f6/raw/coverage-badge.json
    :target: https://unimelbmdap.github.io/collectra/coverage/

.. |torchapp badge| image:: https://img.shields.io/badge/torch-app-B1230A.svg
    :target: https://rbturnbull.github.io/torchapp/
    
.. end-badges

.. start-quickstart

Builds pipelines to extract data from collection images.

Installation
==================================

Install using pip:

.. code-block:: bash

    pip install git+https://github.com/unimelbmdap/collectra.git

.. end-quickstart

Design Requirements 
===============

Collectra caters to two user group:
- Operator
- Builder

User Stories
***************
Operator:

- I want to be able to download a ``.collectra`` file and run it to process a set of files.
- I want to be able to fine tune the machine learning engine attached to a machine learning task
- I want to be able to share the ``.collectra`` file with other operators.
- I want to have a GUI that opens up when I open the workflow to perform the above tasks. All save actions are version controlled.

Builder:

- I want to be able to create/edit a task template and engine, which can be used by operators.
- I want to be able to create/edit workflow by chaining together existing task templates and engines.
- I want to be able to save the workflow as a ``.collectra`` file, which can be used by operators.
- I want to have a GUI that can open any workflow. All save actions are version controlled.

.. image:: img/collectra_wf.png
    :width: 500px
    :alt: Collectra workflow

Architecture
*************
Each project is saved as a ``.collectra`` file, describing the the workflow which includes:

- A task: how to process a particular file
- An engine: what is used to process the file

When a raw file is processed by the task, the output is saved as a ``.grapto`` file.

Both ``.collectra`` and ``.grapto`` files conform to the `RO-Crate 1.1 specification <https://www.researchobject.org/ro-crate/specification/1.2/>`_

A ``.collectra`` file contains the following data:
- Definitions of the task and associated engine. 
- If the engine is a trained model, the location of the model is also specified. 

Input:
- A list of files to be processed, which can be either raw images or existing ``.grapto`` files.

Output:
- A list of ``.grapto`` files, which are the processed outputs of the input files.

Usage:
=======

.. code-block:: bash

    collectra train --workflow grapto.collectra --engine yolo_primary_label --training train --validation valid

 
Credits
==================================

.. start-credits

Robert Turnbull

For more information contact: <robert.turnbull@unimelb.edu.au>

James Quang

For more information contact: <james.quang@unimelb.edu.au>

Created using `torchapp <https://github.com/rbturnbull/torchapp>`_.

.. end-credits

