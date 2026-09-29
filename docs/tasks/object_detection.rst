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

All three detectors expose ``run`` and ``train``. To extract crop images or
evaluate predicted boxes, use the commands on :doc:`../artefacts/image_crop`.
