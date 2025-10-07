import pytest

from unittest.mock import MagicMock, patch
from pathlib import Path


class MockYOLOType(MagicMock):            
    def train(self, **kwargs):
        required_keys = ["data", "project", "device", "epochs", "imgsz"]
        for key in required_keys:
            if key not in kwargs:
                raise ValueError(f"Missing '{key}' parameter") 
        with patch("collectra.tasks.machine_learning.yolo.DetMetrics") as det_metrics_mock:
            det_metrics_instance = MagicMock()
            det_metrics_mock.return_value = det_metrics_instance        
            det_metrics_instance.save_dir = Path(kwargs["project"]) / "runs" / "train" / "exp"
            (det_metrics_instance.save_dir).mkdir(parents=True, exist_ok=True)
            (det_metrics_instance.save_dir / "best.pt").touch()  # Create an empty best.pt file
            det_metrics_instance.results_dict = {"mAP_0.5": 0.85, "mAP_0.5:0.95": 0.65}
            return det_metrics_instance

    def run(self, **kwargs):
        required_keys = ["source", "conf", "iou", "device"]
        for key in required_keys:
            if key not in kwargs:
                raise ValueError(f"Missing '{key}' parameter") 
        with patch("collectra.tasks.machine_learning.yolo.Results") as results_mock:
            results_instance = MagicMock()
            results_mock.return_value = results_instance        
            results_instance.boxes = ["box1", "box2"]  # Mocked boxes
            results_instance.masks = None  # No masks for this mock
            results_instance.keypoints = None  # No keypoints for this mock
            return results_instance

@pytest.fixture
def initalised_yolo_model(classes):        
    with patch("collectra.tasks.machine_learning.yolo.YOLO", new=MockYOLOType) as yolo_mock:        
        yolo_mock_client = MagicMock()     
        yolo_mock.return_value = yolo_mock_client                        
        yield yolo_mock, yolo_mock_client