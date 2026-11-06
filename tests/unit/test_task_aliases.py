"""Renamed tasks keep their former names as aliases."""

import importlib

import pytest
import yaml

import collectra
import collectra.tasks

RENAMED = [
    ("ImageSegmentationSam3", "ImageSegmenterSam3", "image_segmentation.sam3"),
    ("ImageSegmentationRFDETR", "ImageSegmenterRFDETR", "image_segmentation.rfdetr"),
    ("ImageSegmentationYOLO", "ImageSegmenterYOLO", "image_segmentation.yolo"),
    ("ObjectDetectionTask", "ObjectDetector", "object_detection.base"),
    ("ObjectDetectionYOLO", "ObjectDetectorYOLO", "object_detection.yolo"),
    ("ObjectDetectionDETR", "ObjectDetectorDETR", "object_detection.detr"),
    ("ObjectDetectionRFDETR", "ObjectDetectorRFDETR", "object_detection.rfdetr"),
]


@pytest.mark.parametrize("old, new, module", RENAMED)
def test_the_old_name_is_the_same_class_as_the_new_one(old, new, module):
    cls = getattr(collectra, new)
    assert cls.__name__ == new
    assert getattr(collectra, old) is cls
    if new != "ObjectDetector":  # the base class is not in collectra.tasks
        assert getattr(collectra.tasks, new) is cls
        assert getattr(collectra.tasks, old) is cls
    source = importlib.import_module(f"collectra.tasks.{module}")
    assert getattr(source, old) is getattr(source, new) is cls


@pytest.mark.parametrize("old, new, module", RENAMED)
def test_both_names_are_public(old, new, module):
    assert {old, new} <= set(collectra.__all__)
    if new != "ObjectDetector":
        assert {old, new} <= set(collectra.tasks.__all__)


@pytest.mark.parametrize("old, new, module", RENAMED)
def test_a_pipeline_file_can_use_either_name(tmp_path, old, new, module):
    from collectra import Pipeline

    if new == "ObjectDetector":  # the abstract base is not used in a pipeline file
        pytest.skip("not a task to put in a pipeline")

    for index, kind in enumerate([old, new]):
        path = tmp_path / f"pipeline{index}.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "collectra_pipeline_metadata": {
                        "name": "Segments",
                        "ext": "seg",
                        "version": "1",
                    },
                    "segment": {
                        "type": f"collectra.{kind}",
                        "input": "page",
                        "output": "region",
                    },
                }
            )
        )
        task = Pipeline.from_file(path).cli_tasks()["segment"]
        assert type(task) is getattr(collectra, new)
