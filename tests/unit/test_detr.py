from collectra import Image, ImageCrop, ObjectDetectionDETR

TEST_DETR_MODEL = "hf-internal-testing/tiny-random-detr"


def test_train_detr_temp_dir(classes, images, train_yolo, debug, tmpdir):
    """Test training DETR model with temporary directory for logs and weights."""
    try:
        assert len(images) > 0, "No images provided for training. Check fixture"
        detr_task = ObjectDetectionDETR(name="label-detector", model=TEST_DETR_MODEL)
        assert isinstance(detr_task, ObjectDetectionDETR)

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

        # Reuse the same helper fixture used by YOLO tests so setup is identical.
        results = train_yolo(
            detr_task,
            *prepared_images,
            classes=classes,
            project=f"{detr_task.name}-test",
            batch=1,
            wandb=False,
        )
        assert results, "Training failed to return any results"
        assert results.results_dict is not None, "results_dict should exist"
        best_model_path = results.save_dir / "weights" / "best.pt"
        assert best_model_path.exists(), f"best.pt not found in {best_model_path}"
    except Exception as e:
        debug(e)


def test_run_detr(image, debug):
    """Test the DETR object detection model on a single image."""
    try:
        detr_task = ObjectDetectionDETR(name="label-detector", model=TEST_DETR_MODEL)
        assert isinstance(detr_task, ObjectDetectionDETR)
        detections = detr_task.run(image)
        assert isinstance(detections, list), "Detections should be a list"
        assert all(
            isinstance(det, ImageCrop) for det in detections
        ), "All detections should be ImageCrop instances"
    except Exception as e:
        debug(e)
