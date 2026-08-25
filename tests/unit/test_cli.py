from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import cappa
import collectra
import pytest

from collectra.cli import command, group, invoke
from collectra.main import find_pipeline
from collectra.main import main
from collectra.pipelines.base import Collectra


class Detector:
    def __init__(self, calls: list):
        self.calls = calls

    @command
    def run(self, inputs: list[str]):
        """Run object detection."""
        self.calls.append(("detector-run", inputs))

    @command
    def train(
        self,
        inputs: list[str],
        dropout: float = 0.5,
        batch_size: int = 32,
    ):
        """Train object detection."""
        self.calls.append(("detector-train", inputs, dropout, batch_size))


class Classifier:
    @command
    def run(self, inputs: list[str]):
        """Run classification."""

    @command
    def train(self, inputs: list[str], epochs: int = 100):
        """Train classification."""


class Reader:
    @command
    def run(self, inputs: list[str]):
        """Run the reader."""


class RuntimeApplication:
    def __init__(self, tasks: dict[str, object], calls: list | None = None):
        self.tasks = tasks
        self.calls = calls if calls is not None else []

    @command
    def run(self, inputs: list[str]):
        """Run the complete pipeline."""
        self.calls.append(("run", inputs))

    @command
    def install(self, name: str):
        """Install this pipeline."""

    @group
    def task(self):
        """Operate on a task."""
        return self.tasks


def help_for(app, argv, capsys) -> str:
    with pytest.raises(cappa.HelpExit):
        invoke(app, argv, name="collectra")
    return capsys.readouterr().out


def test_dynamic_recursive_help(capsys):
    calls = []
    app = RuntimeApplication(
        {"label_classifier": Classifier(), "label_detector": Detector(calls)}
    )

    root_help = help_for(app, ["--help"], capsys)
    assert all(name in root_help for name in ("run", "install", "task"))

    task_help = help_for(app, ["task", "--help"], capsys)
    assert "label_classifier" in task_help
    assert "label_detector" in task_help

    detector_help = help_for(app, ["task", "label_detector", "--help"], capsys)
    assert "run" in detector_help
    assert "train" in detector_help


def test_capabilities_and_signature_help_are_task_specific(capsys):
    app = RuntimeApplication({"classifier": Classifier(), "reader": Reader()})

    reader_help = help_for(app, ["task", "reader", "--help"], capsys)
    assert "run" in reader_help
    assert "train" not in reader_help

    classifier_help = help_for(app, ["task", "classifier", "train", "--help"], capsys)
    assert "--epochs" in classifier_help


def test_parsed_arguments_reach_bound_method():
    calls = []
    app = RuntimeApplication({"detector": Detector(calls)})
    invoke(
        app,
        [
            "task",
            "detector",
            "train",
            "one",
            "two",
            "--dropout",
            "0.25",
            "--batch-size",
            "16",
        ],
        name="collectra",
    )
    assert calls == [("detector-train", ["one", "two"], 0.25, 16)]


@pytest.mark.parametrize(
    "argv",
    [
        ["task", "missing", "--help"],
        ["task", "reader", "train", "--help"],
    ],
)
def test_unknown_runtime_commands_fail_cleanly(argv):
    app = RuntimeApplication({"reader": Reader()})
    with pytest.raises(cappa.Exit) as error:
        invoke(app, argv, name="collectra")
    assert error.value.code == 2


def test_two_pipelines_produce_different_trees(capsys):
    first = RuntimeApplication({"detector": Detector([])})
    second = RuntimeApplication({"reader": Reader()})
    first_help = help_for(first, ["task", "--help"], capsys)
    second_help = help_for(second, ["task", "--help"], capsys)
    assert "detector" in first_help and "reader" not in first_help
    assert "reader" in second_help and "detector" not in second_help


def test_find_pipeline_accepts_both_option_forms():
    assert find_pipeline(["--pipeline", "a.yaml", "task"])[0] == Path("a.yaml")
    assert find_pipeline(["task", "--pipeline=b.yaml"])[0] == Path("b.yaml")


def write_empty_pipeline(path: Path) -> Path:
    pipeline_file = path / "pipeline.yaml"
    pipeline_file.write_text(
        "collectra_pipeline_metadata:\n"
        "  name: Test\n"
        "  ext: collectra\n"
        "  version: 1.0.0\n"
    )
    return pipeline_file


def test_actual_entry_point_root_help(tmp_path, capsys):
    pipeline_file = write_empty_pipeline(tmp_path)
    with pytest.raises(cappa.HelpExit):
        main(["--pipeline", str(pipeline_file), "--help"])
    output = capsys.readouterr().out
    assert all(name in output for name in ("run", "install", "task"))


def test_installed_pipeline_launcher_delegates_to_main(tmp_path):
    pipeline_file = write_empty_pipeline(tmp_path)
    pipeline = Collectra.from_file(pipeline_file)
    bin_dir = tmp_path / "bin"
    pipeline.cli_install("test-pipeline", bin_dir=bin_dir)
    launcher = bin_dir / "test-pipeline"

    result = subprocess.run(
        [str(launcher), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert all(name in result.stdout for name in ("run", "install", "task"))


def test_cli_import_does_not_import_ml_backends():
    script = (
        "import sys; import collectra.main; "
        "print(*(name in sys.modules for name in "
        "('torch', 'torchvision', 'ultralytics', 'rfdetr', 'transformers')))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "False False False False False"
    assert result.stderr == ""


def test_model_task_class_discovery_does_not_import_backends():
    script = (
        "import sys; "
        "from collectra.tasks.object_detection.yolo import ObjectDetectionYOLO; "
        "from collectra.tasks.image_classifier.yolo import ImageClassifierYOLO; "
        "from collectra.tasks.object_detection.detr import ObjectDetectionDETR; "
        "from collectra.tasks.object_detection.rfdetr import ObjectDetectionRFDETR; "
        "from collectra.tasks.machine_learning.orienters import ImageOrienter; "
        "print(*(name in sys.modules for name in "
        "('torch', 'torchvision', 'ultralytics', 'rfdetr', 'transformers')))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "False False False False False"
    assert result.stderr == ""


def test_pipeline_help_with_model_tasks_does_not_import_backends(tmp_path):
    pipeline_file = tmp_path / "pipeline.yaml"
    pipeline_file.write_text(
        "collectra_pipeline_metadata:\n"
        "  name: Lazy\n"
        "  ext: collectra\n"
        "  version: 1.0.0\n"
        "detector:\n"
        "  type: collectra.ObjectDetectionYOLO\n"
        "  model: model.pt\n"
        "rfdetr_detector:\n"
        "  type: collectra.ObjectDetectionRFDETR\n"
        "  model: model.pth\n"
    )
    script = (
        "import sys; import cappa; from collectra.main import main; "
        f"path = {str(pipeline_file)!r}; "
        "\ntry: main(['--pipeline', path, '--help'])"
        "\nexcept cappa.HelpExit: pass"
        "\nprint('BACKENDS', *(name in sys.modules for name in "
        "('torch', 'torchvision', 'ultralytics', 'rfdetr', 'transformers')))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "BACKENDS False False False False False" in result.stdout
    assert result.stderr == ""


def test_training_is_owned_by_concrete_tasks():
    assert not hasattr(Collectra, "train")
    assert not hasattr(Collectra, "save_train")
    assert "MachineLearningTask" not in collectra.__all__
    assert not hasattr(collectra, "MachineLearningTask")


def test_model_task_classes_have_domain_specific_module_paths():
    from collectra import (
        ImageClassifierYOLO,
        ObjectDetectionDETR,
        ObjectDetectionRFDETR,
        ObjectDetectionYOLO,
    )

    assert ObjectDetectionYOLO.__module__ == "collectra.tasks.object_detection.yolo"
    assert ObjectDetectionDETR.__module__ == "collectra.tasks.object_detection.detr"
    assert ObjectDetectionRFDETR.__module__ == "collectra.tasks.object_detection.rfdetr"
    assert ImageClassifierYOLO.__module__ == "collectra.tasks.image_classifier.yolo"


def test_yolo_task_types_are_siblings():
    from collectra import ImageClassifierYOLO, ObjectDetectionYOLO, YOLOTask

    assert issubclass(ObjectDetectionYOLO, YOLOTask)
    assert issubclass(ImageClassifierYOLO, YOLOTask)
    assert not issubclass(ImageClassifierYOLO, ObjectDetectionYOLO)


def test_concrete_model_tasks_own_typed_train_commands(capsys):
    from collectra import ImageClassifierYOLO, ObjectDetectionDETR

    app = RuntimeApplication(
        {
            "classifier": ImageClassifierYOLO("classifier", model="model.pt"),
            "detector": ObjectDetectionDETR("detector", model="default"),
        }
    )
    classifier_help = help_for(app, ["task", "classifier", "train", "--help"], capsys)
    detector_help = help_for(app, ["task", "detector", "train", "--help"], capsys)

    assert "--imgsz" in classifier_help
    assert "--model-name" not in classifier_help
    assert "--model-name" in detector_help
    assert "--weight-decay" in detector_help
