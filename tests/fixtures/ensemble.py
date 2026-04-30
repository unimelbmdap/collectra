"""Test fixtures for ensemble module testing.

Provides fixtures for:
- Sample boxes for IoU calculations
- Sample texts for clustering tests
- Mock ensemble processor with dependencies
- Temporary folders with collectra result files
"""

from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest
import yaml


@pytest.fixture
def sample_boxes():
    """Create sample boxes for IoU testing."""
    return {
        "box1": [0.0, 0.0, 0.5, 0.5],  # Top-left quadrant
        "box2": [0.25, 0.25, 0.75, 0.75],  # Center overlap
        "box3": [0.5, 0.5, 1.0, 1.0],  # Bottom-right quadrant
        "box4": [0.0, 0.0, 0.5, 0.5],  # Identical to box1
        "box_zero_area": [0.5, 0.5, 0.5, 0.5],  # Zero area
        "box_non_overlapping": [0.8, 0.8, 1.0, 1.0],  # No overlap with box1
    }


@pytest.fixture
def sample_texts():
    """Create sample texts for clustering tests."""
    return {
        "identical": ["hello world", "hello world", "hello world"],
        "similar": ["hello world", "hello worlds", "helo world"],
        "distinct": ["apple", "banana", "cherry"],
        "mixed": ["hello", "hello!", "goodbye", "goodby"],
        "single": ["only one"],
        "empty": [],
    }


@pytest.fixture
def sample_provenance():
    """Create sample provenance lists matching sample_texts."""
    return {
        "identical": ["src1::text1", "src2::text1", "src3::text1"],
        "similar": ["src1::text1", "src2::text1", "src3::text1"],
        "distinct": ["src1::text1", "src1::text2", "src1::text3"],
        "mixed": ["src1::t1", "src2::t1", "src1::t2", "src2::t2"],
        "single": ["src1::text1"],
        "empty": [],
    }


@pytest.fixture
def mock_image_crop():
    """Create a mock ImageCrop object."""

    def _create_crop(
        id_str,
        x_center=0.5,
        y_center=0.5,
        width=0.2,
        height=0.2,
        path_name="image.jpg",
        parents=None,
    ):
        crop = MagicMock()
        crop.id = id_str
        crop.x_center = x_center
        crop.y_center = y_center
        crop.width_relative = width
        crop.height_relative = height
        crop.get_path.return_value.name = path_name
        crop.parents = parents or []
        return crop

    return _create_crop


@pytest.fixture
def mock_text():
    """Create a mock Text object."""

    def _create_text(id_str, data, parents=None):
        text = MagicMock()
        text.id = id_str
        text.data = data
        text.parents = parents or []
        return text

    return _create_text


@pytest.fixture
def mock_data_node():
    """Create a mock DataNode."""

    def _create_node(name, ensemble_items=None):
        node = MagicMock()
        node.name = name
        node.ensemble_items = ensemble_items or {}
        node.ensemble = False
        return node

    return _create_node


@pytest.fixture
def mock_node_manager():
    """Create a mock NodeGraphManager."""
    manager = MagicMock()
    manager.resolve_node = MagicMock(side_effect=lambda n: MagicMock(name=n))
    manager.reset_all_nodes = MagicMock()
    manager.get_task_nodes = MagicMock(return_value=[])
    manager.get_data_nodes = MagicMock(return_value=[])
    manager.get_node_names = MagicMock(return_value=[])
    manager.get_parents = MagicMock(return_value=[])
    manager.get_children = MagicMock(return_value=[])
    manager.get_children_data = MagicMock(return_value=[])
    return manager


@pytest.fixture
def mock_flow():
    """Create a mock workflow flow graph."""
    flow = MagicMock()
    flow.nodes = {}
    flow.predecessors = MagicMock(return_value=[])
    flow.successors = MagicMock(return_value=[])
    return flow


@pytest.fixture
def mock_ensemble_processor(mock_node_manager):
    """Create an EnsembleProcessor with mocked dependencies."""
    from collectra.pipelines.ensemble import EnsembleProcessor

    processor = EnsembleProcessor(
        node_manager=mock_node_manager,
        ext=".collectra",
        name="test_workflow",
        version="1.0.0",
    )
    return processor


@pytest.fixture
def temp_ensemble_folders(tmp_path):
    """Create temporary folders with collectra result files for integration testing."""

    def _create_folders(num_folders=2, files_per_folder=1):
        folders = []
        for i in range(num_folders):
            folder = tmp_path / f"source_{i}"
            folder.mkdir(parents=True, exist_ok=True)

            for j in range(files_per_folder):
                result_folder = folder / f"file_{j}.collectra"
                result_folder.mkdir(parents=True, exist_ok=True)

                # Create results.yaml
                results = {
                    "input_image": {
                        "type": "collectra.Image",
                        "id": f"input_image_src{i}_file{j}",
                        "data": "test_image.jpg",
                    },
                    "detected_region": {
                        "type": "collectra.ImageCrop",
                        "id": f"detected_region_src{i}_file{j}",
                        "data": "test_image.jpg",
                        "parents": f"input_image_src{i}_file{j}",
                        "x_center": 0.5 + (i * 0.01),  # Slight variation
                        "y_center": 0.5 + (i * 0.01),
                        "width_relative": 0.2,
                        "height_relative": 0.2,
                    },
                    "extracted_text": {
                        "type": "collectra.Text",
                        "id": f"extracted_text_src{i}_file{j}",
                        "data": f"Hello World {i}",
                        "parents": f"detected_region_src{i}_file{j}",
                    },
                }

                with open(result_folder / "results.yaml", "w") as f:
                    yaml.dump(results, f)

                # Create a dummy artifact file
                (result_folder / "test_image.jpg").touch()

            folders.append(folder)

        return folders, tmp_path / "output"

    return _create_folders


@pytest.fixture
def sample_ensemble_data():
    """Create sample ensembled data for parent resolution tests."""
    return {
        "input_image": {
            "type": "collectra.Image",
            "id": "input_image_ensemble",
            "data": "test.jpg",
            "ensemble": ["src1::input1", "src2::input1"],
        },
        "detected_regions": [
            {
                "type": "collectra.ImageCrop",
                "id": "detected_region_ensembled_1",
                "ensemble": ["src1::crop1", "src2::crop1"],
                "parents": "input_image_ensemble",
            },
            {
                "type": "collectra.ImageCrop",
                "id": "detected_region_ensembled_2",
                "ensemble": ["src1::crop2", "src2::crop2"],
                "parents": "input_image_ensemble",
            },
        ],
    }


@pytest.fixture
def sample_wbf_input():
    """Create sample input for Weighted Box Fusion testing."""
    return {
        "boxes_per_source": [
            [[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.7, 0.7]],  # Source 1
            [[0.11, 0.11, 0.31, 0.31], [0.51, 0.51, 0.71, 0.71]],  # Source 2 (similar)
        ],
        "scores_per_source": [
            [1.0, 1.0],
            [1.0, 1.0],
        ],
        "labels_per_source": [
            [0, 0],
            [0, 0],
        ],
        "provenance_per_source": [
            ["src1::crop1", "src1::crop2"],
            ["src2::crop1", "src2::crop2"],
        ],
    }
