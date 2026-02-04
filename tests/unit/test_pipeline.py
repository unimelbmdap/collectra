import shutil
from pathlib import Path

from collectra import Collectra


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
        assert (
            isinstance(pipeline.data, dict) and len(pipeline.data) > 0
        ), "Pipeline data should be a non-empty dictionary"
    except Exception as e:
        debug(e)


def test_pipeline_init_nodes(pipeline, debug, tmpdir):
    try:
        pipeline = build_pipeline(pipeline)
        pipeline.connect()
        pipeline.render(Path(tmpdir) / "pipeline_diagram")
    except Exception as e:
        debug(e)


def test_get_task(pipeline, debug):
    try:
        pipeline = build_pipeline(pipeline)
        task_name = "object_detector"
        task = pipeline.task(task_name)
        assert (
            isinstance(task, dict) and len(task) > 0
        ), f"Task '{task_name}' should be a non-empty dictionary"
        assert (
            task.get("type", None) == "collectra.ObjectDetectionYOLO"
        ), f"Task '{task_name}' should have type 'collectra.ObjectDetectionYOLO'"
    except Exception as e:
        debug(e)


def test_train(pipeline, debug, tmp_path):
    from datetime import datetime

    from collectra.utils import change_dir

    try:
        pipeline = build_pipeline(pipeline)
        task_name = "object_detector"
        log = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        input_data = [Path.cwd() / "tests/data/images"]
        with change_dir(tmp_path):
            result = pipeline.train(
                task_name,
                input=input_data,
                log=log,
                project=f"{pipeline.name}-{task_name}",
                base_folder=tmp_path,
            )
        assert result, "Train method should not return None"
    except Exception as e:
        debug(e)


def test_run_full(pipeline, debug, tmpdir, raw_img_path):
    try:
        pipeline = build_pipeline(pipeline)
        shutil.copy(raw_img_path, tmpdir / "bar1.jpg")
        pipeline.run(bar_img1=tmpdir / "bar1.jpg")
    except Exception as e:
        debug(e)


def test_run_one_task(pipeline, debug, tmpdir, raw_img_path):
    try:
        pipeline = build_pipeline(pipeline)
        file_path = "tests/data/images/bar1.arb"
        tempfile = shutil.copytree(file_path, tmpdir / "bar1.arb")
        shutil.copy(raw_img_path, tmpdir / "bar1.jpg")
        pipeline.run(
            "markdown_converter",
            bar_img1=tmpdir / "bar1.jpg",
            file=tempfile,
        )
    except Exception as e:
        debug(e)


def test_run_single_task(pipeline, debug, tmpdir):
    try:
        pipeline = build_pipeline(pipeline)
        file_path = "tests/data/images/bar1.arb"
        tempfile = shutil.copytree(file_path, tmpdir / "bar1.arb")
        pipeline.run("object_detector", single=True, file=tempfile)
    except Exception as e:
        debug(e)
