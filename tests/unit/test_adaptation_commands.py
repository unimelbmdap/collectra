"""CLI routing and validation without optional detector dependencies."""

from pathlib import Path

import pytest

from collectra import ObjectDetectionRFDETR, ObjectDetectionYOLO
from collectra.cli import invoke
from collectra.tasks.object_detection.adaptation_commands import _output_path


@pytest.mark.parametrize(
    "task,command",
    [
        (ObjectDetectionRFDETR, "adapt-input-channels"),
        (ObjectDetectionRFDETR, "adapt-feature-fusion"),
        (ObjectDetectionYOLO, "adapt-feature-fusion"),
    ],
)
def test_command_rejects_invalid_options_before_loading_models(tmp_path, task, command):
    source = tmp_path / "rgb.pt"
    source.write_bytes(b"placeholder")
    args = [command, str(source), "0"]
    with pytest.raises(ValueError, match="num_views"):
        invoke(task("detector"), args)
    assert source.read_bytes() == b"placeholder"


def test_output_defaults_and_overwrite_protection(tmp_path):
    source = tmp_path / "rgb.pth"
    source.write_bytes(b"input")
    output = _output_path(source, None, 5, "15ch", False)
    assert output == tmp_path / "rgb-5views-15ch.pth"
    with pytest.raises(ValueError, match="differ"):
        _output_path(source, source, 5, "15ch", True)
    output.write_bytes(b"existing")
    with pytest.raises(ValueError, match="already exists"):
        _output_path(source, output, 5, "15ch", False)
    assert _output_path(source, output, 5, "15ch", True) == output
    with pytest.raises(ValueError, match="does not exist"):
        _output_path(tmp_path / "missing.pth", None, 5, "15ch", False)
