Object Detection
================

Object detectors consume an ``Image`` and produce labelled ``ImageCrop``
artefacts. Their output names should match the labels in the trained model.
Use annotated crops with source-image relationships for training.

.. toctree::
   :maxdepth: 1

   object_detection_yolo
   object_detection_rfdetr
   object_detection_detr
   object_detection_sam3

Shared settings
---------------

The detectors inherit from ``collectra.ObjectDetector``, which
declares the settings they have in common. Give them in the task's
``pipeline.yaml`` entry; ``run`` and ``train`` use them unless an option such as
``--epochs`` or ``--resolution`` is given for that command.

.. code-block:: yaml

    label_detector:
      type: collectra.ObjectDetectorYOLO
      model: label_detector.pt
      input: specimen_sheet
      output: [primary_label, handwritten_data]
      resolution: 1280
      threshold: 0.3
      epochs: 200

``resolution``
    Image size in pixels (default 1280). Training always uses it. Running uses
    it only when you set it, so a model otherwise keeps the size it was trained
    with.
``threshold``
    Minimum confidence for a detection, from 0 to 1 (default 0.25).
``epochs``, ``batch``, ``learning_rate``, ``weight_decay``
    Training settings. Each backend keeps the default that suits its library
    (for example ``batch`` is 16 for YOLO, 4 for RF-DETR and 2 for DETR).

``orient``
    Output classes whose crops are oriented after detection (default none).
    Each crop of these classes goes through an orientation model, as
    :doc:`image_orientation` does, and is saved facing the predicted way up.
    Crops of other classes keep the orientation of their image. Every name must
    be one of the detector's outputs.
``orienter_model``
    The orientation checkpoint used for ``orient``, relative to the pipeline
    folder. Without it, the published model of
    `deep-image-orientation-detection
    <https://github.com/duartebarbosadev/deep-image-orientation-detection>`_
    is downloaded from Hugging Face on first use and cached.

.. code-block:: yaml

    component_detector:
      type: collectra.ObjectDetectorRFDETR
      model: component_detector.pt
      input: specimen_sheet
      output: [primary_label, stamp, swatch, scale]
      orient: [primary_label, stamp, scale]

Settings only one backend has, such as YOLO's ``iou``, are listed in its own
page. A setting a detector does not know is reported with a warning, naming the
settings it does understand. The old name ``imgsz`` and YOLO's ``lr0`` are still
accepted, with a warning, as ``resolution`` and ``learning_rate``.

To write your own detector, subclass ``ObjectDetector`` and implement
``_init_model``, ``run`` and ``train``. ``run`` should return its crops through
``self._orient_crops(crops)`` so that ``orient`` applies. A detector that can predict several
images at once also implements ``_predict_batch``, and ``prefetch`` then works
for it without more code.

All the detectors expose ``run`` and ``train``. To extract crop images or
evaluate predicted boxes, use the commands on :doc:`../artefacts/image_crop`.
