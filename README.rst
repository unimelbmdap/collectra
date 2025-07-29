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

Collectra caters to two user groups:

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

Architecture
*************

.. image:: img/collectra_wf.png
    :width: 500px
    :alt: Collectra workflow

`Collectra top-level architecture diagram <https://excalidraw.com/#json=KYcVyuFJIo4Jk7dngu3bV,NMmp8pk4D0VGQr7l2KgR3A>`_

A file created by the Collectra application is an RO-Crate compliant (`RO-Crate 1.1 specification <https://www.researchobject.org/ro-crate/specification/1.1/>`_) file that describes a workflow. It has the following structure:

.. code-block:: json

  {
        "@context": "https://w3id.org/ro/crate/1.1/context",
        "@graph": [
            {
                "@id": "./",
                "@type": "Dataset",
                "datePublished": "2025-07-29T08:16:24+00:00",
                "file_format": "hespi",
                "hasPart": [                    
                    {
                        "@id": "648bfb1e-bcf0-41e4-8eed-3ce7c912a6b6"
                    },
                    {
                        "@id": "best.pt-training-params"
                    },
                    {
                        "@id": "best.pt"
                    }
                ]
            },
            {
                "@id": "ro-crate-metadata.json",
                "@type": "CreativeWork",
                "about": {
                    "@id": "./"
                },
                "conformsTo": {
                    "@id": "https://w3id.org/ro/crate/1.1"
                }
            },            
            {
                "@id": "648bfb1e-bcf0-41e4-8eed-3ce7c912a6b6",
                "@type": "Task",
                "engine": [
                    {
                        "@id": "best.pt"
                    }
                ],
                "task_type": "detect_object"
            },
            {
                "@id": "best.pt-training-params",
                "@type": "TrainingParameters",
                "device": "cpu",
                "epochs": 1,
                "imgsz": 640,
                "verbose": true
            },
            {
                "@id": "best.pt",
                "@type": "File",
                "engine_type": "yolo",
                "name": "best.pt",
                "trainingParameters": [
                    {
                        "@id": "best.pt-training-params"
                    }
                ]
            }
        ]
    }

All defined task will be saved in the workflow, with the following properties: 

- task_type: how to process a particular file
- engine: what is used to process the file

When a raw file is processed by the task, the output is saved as a ``.something`` file where .something is the file extension defined by the the workflow editor and user

Usage:
=======

Builder
********

Make a new workflow

.. code-block:: bash

    collectra make wf1 -t detect_object,yolo,yolo11n.pt -f hespi

- ``-f`` is the file format of the workflow, which is used to determine how to process the files in the workflow
- ``-t`` is the task, which is used to determine how to process the files in the workflow. It must in the format of task_type,engine_type,engine_file_path

Train an ML task

.. code-block:: bash

    collectra train wf1.collectra -t detect_object -i test_data

- ``-t`` is the task type to train, which must be defined in the workflow
- ``-i`` is the input data to train the task
- By default, the logs are saved to output/logs.txt

Compile and build the workflow

 
Credits
==================================

.. start-credits

Robert Turnbull

For more information contact: <robert.turnbull@unimelb.edu.au>

James Quang

For more information contact: <james.quang@unimelb.edu.au>

Created using `torchapp <https://github.com/rbturnbull/torchapp>`_.

.. end-credits

