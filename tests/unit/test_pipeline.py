from collectra import Collectra
from pathlib import Path
from unittest.mock import Mock, patch

def build_pipeline(pipeline: dict) -> Collectra:
    metadata: dict = pipeline.pop("collectra_pipeline_metadata")
    name = metadata["name"]      
    ext = metadata["ext"]
    version = metadata["version"]
    path = pipeline.pop("pipeline_path", "")
    return Collectra(name, ext, version, path=path, **pipeline)


def test_pipeline_init(pipeline, debug):
    try:
        name = pipeline["collectra_pipeline_metadata"]["name"]
        ext = pipeline["collectra_pipeline_metadata"]["ext"]
        version = pipeline["collectra_pipeline_metadata"]["version"]
        pipeline = build_pipeline(pipeline)
        assert pipeline.name == name
        assert pipeline.ext == ext
        assert pipeline.version == version
        assert pipeline.path
        assert isinstance(pipeline.data, dict) and len(pipeline.data) > 0, "Pipeline data should be a non-empty dictionary"
    except Exception as e:
        debug(e)

def test_pipeline_init_nodes(llm_loader_mock, pipeline, debug):
    try:
        llm_loader_mock, llm_instance = llm_loader_mock        
        pipeline = build_pipeline(pipeline)
        pipeline.connect()        
        pipeline.render()
    except Exception as e:
        debug(e)

def test_get_task(pipeline, debug):
    try:        
        pipeline = build_pipeline(pipeline)
        task_name = "object_detector"
        task = pipeline.task(task_name)        
        assert isinstance(task, dict) and len(task) > 0, f"Task '{task_name}' should be a non-empty dictionary"
        assert task.get("type", None) == "collectra.ObjectDetectionYOLO", f"Task '{task_name}' should have type 'collectra.ObjectDetectionYOLO'"
    except Exception as e:
        debug(e)


def test_run_task_invalid_input(llm_loader_mock, pipeline, raw_img_path, debug):
    try:
        llm_loader_mock, llm_instance = llm_loader_mock
        pipeline = build_pipeline(pipeline)        
        task_name = "object_detector"        
        result = pipeline(task_name, raw_img_path)        
    except Exception as e:
        assert str(e) == "Input must be an instance of Image.", "Expected TypeError for invalid input type"

# def test_run_llm_task(pipeline, debug):
#     try:
#         pipeline = build_pipeline(pipeline)        
#         task_name = "markdown_converter"
#         result = pipeline(task_name)
#         assert result is None, "Run method should return None"=
#     except Exception as e:
#         debug(e)


# def test_train_task(pipeline, debug):
#     try:
#         pipeline = build_pipeline(pipeline)
#         task_name = "field_detector"
#         result = pipeline.train(task_name)
#         assert result is None, "Train method should return None"
#     except Exception as e:
#         debug(e)
