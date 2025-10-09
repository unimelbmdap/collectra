import pytest

from pathlib import Path
from unittest.mock import MagicMock, patch

from collectra.models.yolo import YOLOModel
from collectra.types.images import Image

# @pytest.fixture()
# def mock_yolo_model():
#     with patch("collectra.models.yolo.YOLO") as yolo_mock:
#         yolo_mock_client = MagicMock()
#         yolo_mock.return_value = yolo_mock_client
#         yolo_mock_client.predict.return_value = ["results"]
#         yield yolo_mock, yolo_mock_client

# @pytest.fixture()
# def mock_image():
#     with patch("collectra.types.images.Image") as image_mock:
#         image_instance = MagicMock()
#         image_mock.return_value = image_instance
#         yield image_instance

# def test_yolo_model_initialization(mock_yolo_model, model_path):
#     yolo_mock, _ = mock_yolo_model
#     model = YOLOModel(model_path)
#     yolo_mock.assert_called_once_with(model_path, verbose = True)
#     assert model.get_path() == Path(model_path), "Model path should match the provided path"

# def test_yolo_model_run(model_path, mock_yolo_model):
#     yolo_mock, yolo_mock_client = mock_yolo_model
#     key = "test_image1"
#     model = YOLOModel(model_path)
#     test_image1 = Image(Path("tests/data/specimen.jpg"))
#     detections = model.detect(test_image1=test_image1) # argument key should be same as variable key

#     yolo_mock.assert_called_once_with(model_path, verbose = True)
#     yolo_mock_client.predict.assert_called_once()
#     assert isinstance(detections, dict), "Detections should be a dictionary"
#     assert "image" in detections[key], "Detections should contain 'image' key"
#     assert "results" in detections[key], "Detections should contain 'results' key"
