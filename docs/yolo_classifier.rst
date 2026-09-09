YOLO image classification
=========================

``ImageClassifierYOLO`` supports multichannel TIFFs during training and inference,
including cropped images. No channel option is needed: Collectra inspects every
prepared training and validation image and rejects inconsistent channel counts
with filenames and detected counts. The dataset determines the model's input
channels; Ultralytics constructs the model and transfers compatible weights.

TIFF axes metadata determines channel order, supporting channel-first and
channel-last layouts (``CYX``, ``YXC``, ``SYX``, ``YXS``), grayscale images, and
unlabelled page stacks. Ambiguous multidimensional TIFFs are rejected. TIFF
crops retain all bands and their original dtype. Whole TIFFs are copied intact.

TIFFs and non-RGB datasets use a custom Ultralytics classification trainer and
tensor loader. Unsigned integer samples are divided by their dtype's maximum
(255 for uint8, 65535 for uint16); floating-point values are retained as float32.
Normalization is identity: RGB statistics are not applied to spectral bands.
Training uses random resized crops, ``--fliplr``, ``--flipud``, and ``--erasing``.
RGB colour jitter and ``--auto-augment`` are skipped on this path, with a log
message explaining the policy. Validation and inference use resize and centre
crop. RGB sample previews are skipped, while metrics and checkpoints remain
available. Ordinary three-channel non-TIFF datasets retain the standard
Ultralytics classification path.

Use ``ImageClassifierYOLO`` to run saved multichannel checkpoints: it applies
the saved tensor transforms and bypasses Ultralytics' RGB prediction loader.
Calling a checkpoint directly through the stock Ultralytics classification CLI
does not enable this custom data path. Inference requires an input with the
same channel count as the model.

The custom trainer is tested against Ultralytics 8.4.78 on CPU. GPU and
distributed execution have not been exercised by the unit tests. No additional
dependencies are required beyond Collectra's existing TIFF and PyTorch packages.
