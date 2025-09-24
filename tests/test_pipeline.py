import pytest
from pathlib import Path
from typer.testing import CliRunner
from collectra.main import app


class TestPipeline:

    @pytest.fixture(autouse=True)
    def setup(self, tmpdir):
        """Setup method that runs before each test method."""
        self.runner = CliRunner()
        self.dummy_workflow: str = f"{tmpdir}/dummy_workflow"

    def call(self, command: str):
        """Helper method to invoke CLI commands."""
        return self.runner.invoke(app, command.split())

    def make_workflow(self, dummy_workflow: str, format: str):
        """Helper method to create a dummy workflow."""
        return self.call(f"make -f {format} -w {dummy_workflow}")

    def test_pipeline_making(self):
        """Test that pipeline creation works correctly."""
        result = self.make_workflow(self.dummy_workflow, "hespi")
        assert result.exit_code == 0, "Pipeline should be created successfully"
        assert Path(
            f"{self.dummy_workflow}.collectra"
        ).exists(), "Pipeline file should exist with suffix .collectra"

    def test_pipeline_rendering(self):
        self.make_workflow(self.dummy_workflow, "hespi")
        result = self.call(f"render -w {self.dummy_workflow}.collectra")
        assert result.exit_code == 0, "Rendering should be successful"
        assert (
            "Finished rendering" in result.output
        ), "Output should contain rendering message"
        self.call(f"make -w {self.dummy_workflow} -f hespi --as-dir")
        result = self.call(f"render -w {self.dummy_workflow}")
        assert (
            result.exit_code == 0
        ), "Rendering should be successful for directory workflow"
        assert (
            "Finished rendering" in result.output
        ), "Output should contain rendering message for directory workflow"

    def test_pipeline_add_task(self):
        """Test adding a task to the pipeline."""
        self.make_workflow(self.dummy_workflow, "hespi")
        result = self.call(
            f"add -w {self.dummy_workflow}.collectra -t label_detect,ObjectDetectionYOLO,yolo11n.pt"
        )
        assert result.exit_code == 0, "Adding task should be successful"
        assert (
            "Task with ID label_detect added to the workflow" in result.output
        ), "Output should confirm task addition"
        result = self.call(
            f"add -w {self.dummy_workflow}.collectra -t label_detect,ObjectDetectionYOLO,yolo11n.pt"
        )
        assert result.exit_code == 0, "Adding duplicate task should be successful"
        assert (
            "Task name is empty or already exists" in result.output
        ), "Output should confirm duplicate task handling"

    # def test_pipeline_train_task(self):
    #     """Test training a task in the pipeline."""
    #     self.make_workflow(self.dummy_workflow, "hespi")
    #     result = self.call(f"add -w {self.dummy_workflow}.collectra -t label_detect,ObjectDetectionYOLO,yolo11n.pt")
    #     assert result.exit_code == 0, "Adding task should be successful"
    #     result = self.call(f"train -w {self.dummy_workflow}.collectra -t label_detect")
    #     assert result.exit_code == 0, "Training task should be successful"
    #     assert "Training task label_detect completed successfully" in result.output, "Output should confirm task training completion"
