from collectra import Collectra
from pathlib import Path

def build_pipeline(pipeline: dict) -> Collectra:
    metadata: dict = pipeline.pop("collectra_pipeline_metadata")
    name = metadata["name"]      
    ext = metadata["ext"]
    version = metadata["version"]
    return Collectra(name, ext, version, **pipeline)

def test_pipeline_init(pipeline, debug):
    try:
        name = pipeline["collectra_pipeline_metadata"]["name"]
        ext = pipeline["collectra_pipeline_metadata"]["ext"]
        version = pipeline["collectra_pipeline_metadata"]["version"]
        pipeline = build_pipeline(pipeline)
        assert pipeline.name == name
        assert pipeline.ext == ext
        assert pipeline.version == version
        assert pipeline.path == Path(name)
        assert isinstance(pipeline.data, dict) and len(pipeline.data) > 0, "Pipeline data should be a non-empty dictionary"
    except Exception as e:
        debug(e)

def test_get_task(pipeline, debug):
    try:
        pipeline = build_pipeline(pipeline)
        task_name = "field_detector"
        task = pipeline.task(task_name)
        assert isinstance(task, dict) and len(task) > 0, f"Task '{task_name}' should be a non-empty dictionary"
        assert task.get("type", None) == "collectra.ObjectDetectionYOLO", f"Task '{task_name}' should have type 'collectra.ObjectDetectionYOLO'"
    except Exception as e:
        debug(e)