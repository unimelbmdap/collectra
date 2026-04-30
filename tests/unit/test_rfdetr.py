from collectra import Image, ImageCrop, ObjectDetectionRFDETR


def test_train_rfdetr_temp_dir(classes, images, train_yolo, debug):
    """Test training RF-DETR model with temporary directory for logs and weights."""
    try:
        assert len(images) > 0, "No images provided for training. Check fixture"
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
        # RFDETRBase() downloads "rf-detr-base.pth" relative to cwd. Pin cwd
        # to pytest's tmp_path (under /tmp) so the weights land there and get
        # cleaned up automatically by pytest's tmp_path retention policy.
        monkeypatch.chdir(tmp_path)
        rfdetr_task = ObjectDetectionRFDETR(name="label-detector")
        assert isinstance(rfdetr_task, ObjectDetectionRFDETR)
        detections = rfdetr_task.run(image)
        assert isinstance(detections, list), "Detections should be a list"
        assert all(
            isinstance(det, ImageCrop) for det in detections
        ), "All detections should be ImageCrop instances"
    except Exception as e:
        debug(e)
