Image Classification
====================

Image classifiers consume images or crops and return :doc:`../artefacts/link`
artefacts under the predicted class name. Each Link retains its source image
or crop. Configure the class names in the task's ``output`` list.

.. toctree::
   :maxdepth: 1

   ../yolo_classifier
   ../torchvision_classifier
   ../huggingface_classifier

Each classifier exposes ``run`` and ``train``. Training uses labelled Links in
Collectra result directories; see :doc:`../quickstart` for an annotated example.
Torchvision and Hugging Face classifiers support ``--wandb`` as an explicit
training opt-in.
