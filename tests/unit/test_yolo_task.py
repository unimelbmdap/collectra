from collectra import ObjectDetectionYOLO
from ultralytics.utils.metrics import DetMetrics
from pathlib import Path

def test_train_yolo(images, classes, initalised_yolo_model):
    yolo_mock, yolo_mock_client = initalised_yolo_model
    train_images, validation_images = images
    assert len(train_images) > 0, "No images provided for training. Check fixture"
    assert (
        len(validation_images) > 0
    ), "No images provided for validation. Check fixture"
    yolo_task = ObjectDetectionYOLO(name="label-detector", model="yolo11n.pt")
    assert isinstance(yolo_task, ObjectDetectionYOLO)
    assert isinstance(yolo_task.model, str) or isinstance(
        yolo_task.model, Path
    ), "Model should be a string or path initially"
    results = yolo_task.train(
        train_img=train_images,
        val_img=validation_images,
        classes=classes,
    )
    assert results is not None, "Training failed to return any results"
    assert results, "Training should return results"    
    assert (results.save_dir / "best.pt").exists(), "best.pt not found in save_dir"    
    assert results.results_dict is not None, "results_dict should exist"    

# def test_run_yolo(image, initalised_yolo_model): 
#     yolo_task = ObjectDetectionYOLO(name="label-detector", model="yolo11n.pt")
#     assert isinstance(yolo_task, ObjectDetectionYOLO)    
#     detections = yolo_task.run(image)
#     assert isinstance(detections, list), "Detections should be a list"    