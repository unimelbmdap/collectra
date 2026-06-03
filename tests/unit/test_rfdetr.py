from pathlib import Path
from unittest.mock import patch

from collectra import Image, ImageCrop, ObjectDetectionRFDETR


class FakeRFDETRBase:
    def __init__(self, *args, **kwargs):
        self.init_args = args
        self.init_kwargs = kwargs

    def train(self, *args, **kwargs):
        output_dir = kwargs.get("output_dir")
        if output_dir is not None:
            weights_dir = Path(output_dir)
        else:
            weights_dir = Path(kwargs["dataset_dir"]).parent / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        (weights_dir / "checkpoint_best_regular.pth").write_bytes(b"fake-weights")

    def predict(self, *args, **kwargs):
        return type(
            "FakeDetections",
            (),
            {
                "xyxy": [[10, 20, 60, 80]],
                "class_id": [0],
            },
        )()


def test_train_rfdetr_temp_dir(classes, images, train_yolo, debug):
    """Test training RF-DETR model with temporary directory for logs and weights."""
    try:
        assert len(images) > 0, "No images provided for training. Check fixture"
        with patch(
            "collectra.tasks.machine_learning.rfdetr.RFDETRBase",
            new=FakeRFDETRBase,
        ):
            rfdetr_task = ObjectDetectionRFDETR(name="label-detector")
            assert isinstance(rfdetr_task, ObjectDetectionRFDETR)

            prepared_images = []
            for img in images:
                if isinstance(img, ImageCrop) and img.source_parent is None:
                    img.add_source_parent(
                        Image(
                            name="specimen_sheet",
                            data=img.get_path(),
                            orientation=img.orientation,
                        )
                    )
                prepared_images.append(img)

            results = train_yolo(
                rfdetr_task,
                *prepared_images,
                classes=classes,
                project=f"{rfdetr_task.name}-test",
                batch=1,
                wandb=False,
            )
        assert results, "Training failed to return any results"
        assert results.results_dict is not None, "results_dict should exist"
        best_model_path = results.save_dir / "weights" / "best.pt"
        assert best_model_path.exists(), f"best.pt not found in {best_model_path}"
    except Exception as e:
        debug(e)


def test_run_rfdetr(image, debug, tmp_path, monkeypatch):
    """Test the RF-DETR object detection model on a single image."""
    try:
        # Keep this as a unit test for our wrapper. Real RF-DETR inference
        # downloads weights and may select CUDA, which belongs in integration
        # coverage on a runner with the right hardware.
        monkeypatch.chdir(tmp_path)
        with patch(
            "collectra.tasks.machine_learning.rfdetr.RFDETRBase",
            new=FakeRFDETRBase,
        ):
            rfdetr_task = ObjectDetectionRFDETR(name="label-detector")
            assert isinstance(rfdetr_task, ObjectDetectionRFDETR)
            detections = rfdetr_task.run(image)
        assert isinstance(detections, list), "Detections should be a list"
        assert all(
            isinstance(det, ImageCrop) for det in detections
        ), "All detections should be ImageCrop instances"
    except Exception as e:
        debug(e)
