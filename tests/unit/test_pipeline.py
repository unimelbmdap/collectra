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
#         assert (
#             isinstance(pipeline.data, dict) and len(pipeline.data) > 0
#         ), "Pipeline data should be a non-empty dictionary"
#     except Exception as e:
#         debug(e)


# def test_pipeline_init_nodes(llm_loader_mock, pipeline, debug, tmpdir):
#     try:
#         llm_loader_mock, llm_instance = llm_loader_mock
#         pipeline = build_pipeline(pipeline)
#         pipeline.connect()
#         pipeline.render(Path(tmpdir) / "pipeline_diagram")
#     except Exception as e:
#         debug(e)


# def test_get_task(pipeline, debug):
#     try:
#         pipeline = build_pipeline(pipeline)
#         task_name = "object_detector"
#         task = pipeline.task(task_name)
#         assert (
#             isinstance(task, dict) and len(task) > 0
#         ), f"Task '{task_name}' should be a non-empty dictionary"
#         assert (
#             task.get("type", None) == "collectra.ObjectDetectionYOLO"
#         ), f"Task '{task_name}' should have type 'collectra.ObjectDetectionYOLO'"
#     except Exception as e:
#         debug(e)


# def test_train(llm_loader_mock, pipeline, debug, tmpdir):
#     try:
#         llm_loader_mock, llm_instance = llm_loader_mock
#         pipeline = build_pipeline(pipeline)
#         task_name = "object_detector"
#         log_dir = Path(tmpdir) / "train_logs"
#         log_dir.mkdir(parents=True, exist_ok=True)
#         result, validation_result  = pipeline.train(task_name, input=["tests/data/images"], log_dir=log_dir)                
#         assert result, "Train method should not return None"
#         assert validation_result, "Validation result should not be None"
#     except Exception as e:
#         debug(e)    

# def test_run(pipeline, debug, tmpdir, raw_img_path):
#     try:
#         # llm_loader_mock, llm_instance = llm_loader_mock
#         pipeline = build_pipeline(pipeline)
#         log_dir = Path(tmpdir) / "run_logs"
#         log_dir.mkdir(parents=True, exist_ok=True)
#         pipeline.run(log_dir=log_dir, bar_img1=raw_img_path)        
#     except Exception as e:
#         debug(e)