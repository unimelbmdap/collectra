from pathlib import Path

from collectra import ObjectDetectionYOLO, ImageCrop


def test_train_yolo_temp_dir(classes, images, model, train_yolo, debug):
    """Test training YOLO model with temporary directory for logs and weights.

    This test verifies that the YOLO training process completes successfully,
    saves the best model weights, and returns a results dictionary.

    The test results are saved in a temporary directory which is cleaned up after the test.
    If the test fails, the temporary directory is retained for debugging purposes.

    Args:
        images (tuple): A tuple containing training and validation image paths.
        classes (list): A list of class names for object detection.

    """
    import shutil

    try:
        assert len(images) > 0, "No images provided for training. Check fixture"
        yolo_task = ObjectDetectionYOLO(name="label-detector", model=model)
        assert isinstance(yolo_task, ObjectDetectionYOLO)
        assert isinstance(yolo_task.model, str) or isinstance(
            yolo_task.model, Path
        ), "Model should be a string or path initially"
        results, _, log_dir = train_yolo(
            yolo_task,
            *images,
            classes=classes,
            project=f"{yolo_task.name}-test",
        )
        assert results, "Training failed to return any results"
        assert results.results_dict is not None, "results_dict should exist"
        best_model_path = results.save_dir / "weights" / "best.pt"
        assert best_model_path.exists(), f"best.pt not found in {best_model_path}"
    except Exception as e:
        debug(e)


def test_run_yolo(image, model, debug):
    """Test the YOLO object detection model on a single image.

    This test verifies that the YOLO model can process an image and return
    a list of detected objects as ImageCrop instances.

    Args:
        image (Path): Path to the image file to be tested.
        model (str or Path): Path to the YOLO model weights.

    """
    try:
        yolo_task = ObjectDetectionYOLO(name="label-detector", model=model)
        assert isinstance(yolo_task, ObjectDetectionYOLO)
        detections = yolo_task.run(image)
        assert isinstance(detections, list), "Detections should be a list"
        assert all(
            isinstance(det, ImageCrop) for det in detections
        ), "All detections should be ImageCrop instances"
    except Exception as e:
        debug(e)
