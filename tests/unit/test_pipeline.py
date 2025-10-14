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


# def test_pipeline_init(pipeline, debug):
#     try:
#         name = pipeline["collectra_pipeline_metadata"]["name"]
#         ext = pipeline["collectra_pipeline_metadata"]["ext"]
#         version = pipeline["collectra_pipeline_metadata"]["version"]
#         pipeline = build_pipeline(pipeline)
#         assert pipeline.name == name
#         assert pipeline.ext == ext
#         assert pipeline.version == version
#         assert pipeline.path
#         assert isinstance(pipeline.data, dict) and len(pipeline.data) > 0, "Pipeline data should be a non-empty dictionary"
#     except Exception as e:
#         debug(e)

@patch('collectra.tasks.llms.llmloader.load')
def test_pipeline_init_nodes(mock_llm_load, pipeline, debug):
    try:
        # Mock the llmloader.load to return a fake LLM object
        mock_llm = Mock()
        mock_llm.invoke = Mock(return_value="mock response")
        mock_llm_load.return_value = mock_llm
        
        pipeline = build_pipeline(pipeline)
        pipeline.connect()        
        pipeline.render()
    except Exception as e:
        debug(e)

# def test_get_task(pipeline, debug):
#     try:
#         pipeline = build_pipeline(pipeline)
#         task_name = "object_detector"
#         task = pipeline.task(task_name)        
#         assert isinstance(task, dict) and len(task) > 0, f"Task '{task_name}' should be a non-empty dictionary"
#         assert task.get("type", None) == "collectra.ObjectDetectionYOLO", f"Task '{task_name}' should have type 'collectra.ObjectDetectionYOLO'"
#     except Exception as e:
#         debug(e)


# def test_run_task(pipeline, raw_img_path, debug):
#     try:
#         pipeline = build_pipeline(pipeline)        
#         task_name = "object_detector"        
#         result = pipeline(task_name, bar_img1=raw_img_path)
#         assert result is None, "Run method should return None"
#     except Exception as e:
#         debug(e)

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
