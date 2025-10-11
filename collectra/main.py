"""Main command-line interface for the Collectra workflow management system.

This module provides the CLI commands for creating, rendering, training, and running
Collectra workflows. It serves as the entry point for the application and handles
user interactions through the Typer framework.

The module supports the following operations:
    - Creating new workflow configurations
    - Rendering workflow diagrams
    - Training machine learning tasks within workflows
    - Executing complete workflows or specific tasks

Example:
    $ collectra make --workflow my_workflow --version 1.0
    $ collectra run --workflow pipeline.yaml --task detection
"""

from datetime import datetime
from pathlib import Path
from rich import print
from typer import Typer, Option, Argument, Context
from typing_extensions import Annotated
from collectra.pipelines.managers import CollectraManager
from collectra.utils import success_msg, error_msg

import os, shutil, logging, sys

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout)

app = Typer()


@app.command()
def make(
    workflow_path: Annotated[
        str, Option("--workflow", "-w", help="name of the workflow")
    ],
    version: Annotated[str, Option("--version", "-v", help="version of the workflow")],
    format: Annotated[
        str,
        Option(
            "--file-format",
            "-f",
            help="File format for the workflow, e.g., grapto, json, yaml",
        ),
    ] = "",
    as_dir: Annotated[
        bool,
        Option("--as-dir", help="Create the workflow as a directory instead of a file"),
    ] = False,
):
    """Create a new Collectra workflow.

    Creates a new workflow configuration with the specified parameters and saves it
    to the filesystem. The workflow can be created as a file or directory structure
    depending on the as_dir parameter.

    Args:
        workflow_path (str): Name of the workflow to create.
        version (str): Version identifier for the workflow.
        format (str, optional): File format for the workflow (e.g., grapto, hespi).
            Defaults to empty string.
        as_dir (bool, optional): Whether to create the workflow as a directory
            structure instead of a single file. Defaults to False.

    Raises:
        Exception: If the workflow cannot be created due to invalid parameters
            or filesystem errors.
    """
    try:
        metadata = {
            "name": Path(workflow_path).name,
            "version": version,
            "description": "",
            "format": format,
            "out_dir": workflow_path,
            "as_dir": as_dir,
        }
        manager = CollectraManager()
        manager.build(metadata=metadata)
        manager.save()
        workflow = manager.get_pipeline()
        print(success_msg(f"Workflow '{workflow}' created at {workflow.name}"))
    except Exception as e:
        print(error_msg(f"Failed to create workflow: {e}"))


@app.command()
def render(
    workflow: Annotated[Path, Option("-w", "--workflow", help="path to workflow")],
    raw: Annotated[bool, Option(help="print raw pipeline if set")] = False,
):
    """Render the Collectra workflow as a visual diagram.

    Generates an SVG visualization of the workflow showing tasks and their
    dependencies. The diagram can be rendered in raw format (showing the
    underlying graph structure) or formatted view (showing inputs/outputs).

    Args:
        workflow (Path): Path to the workflow configuration file to render.
        raw (bool, optional): Whether to render the raw pipeline graph structure
            instead of the formatted view. Defaults to False.

    Raises:
        Exception: If the workflow file cannot be loaded or rendered due to
            invalid format or missing dependencies.
    """
    try:
        manager = CollectraManager()
        manager.load(workflow)
        manager.get_pipeline().render(raw=raw)
    except Exception as e:
        print(error_msg(f"{e}"))


@app.command()
def train(
    workflow: Annotated[Path, Option("--workflow", "-w", help="path to workflow")],
    task: Annotated[str, Option("--task", "-t", help="task to train")],
    input_files: Annotated[list[str], Argument(help="Input directory of files")],
    keep_log: Annotated[
        bool, Option("--keep-log", help="Keep previous log files")
    ] = False,
):
    """Train a specific machine learning task in the Collectra workflow.

    Executes the training process for a specified task within the workflow,
    using the provided input files for training data. Generates timestamped
    log files during training and optionally cleans them up afterward.

    Args:
        workflow (Path): Path to the workflow configuration file containing the task.
        task (str): Name of the specific task to train within the workflow.
        input_files (list[str]): List of input file paths or directories containing
            training data for the task.
        keep_log (bool, optional): Whether to preserve training log files after
            completion. Defaults to False (logs are deleted).

    Raises:
        Exception: If the task cannot be trained due to invalid task name,
            missing input files, or training process failures.
    """
    try:
        log = f"{task}_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        config = {"input_files": input_files, "output_log": log}
        manager = CollectraManager()
        manager.load(workflow)
        manager.get_pipeline().train(task_name=task, config=config)
        manager.save()
        if not keep_log:
            shutil.rmtree(log, ignore_errors=True)
            log_cache = Path(f"{log}.cache")
            if log_cache.exists():
                os.remove(log_cache)
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(error_msg(f"Failed to train task: {e}"))


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def run(
    workflow: Annotated[Path, Option("--workflow", "-w", help="path to workflow")],
    ctx: Context,
    task: Annotated[str, Option("--task", "-t", help="task to run")] = "",
    output: Annotated[
        Path | None, Option("--output", "-o", help="output directory")
    ] = None,
    force: bool = False,
):
    """Execute a Collectra workflow or specific task within a workflow.

    Runs the specified workflow starting from either a specific task or from
    the root tasks if no task is specified. Additional command-line arguments
    are parsed and passed to the workflow execution as input parameters.

    Args:
        workflow (Path): Path to the workflow configuration file to execute.
        ctx (Context): Typer context containing additional command-line arguments.
        task (str, optional): Name of the specific task to run. If empty, runs
            from root tasks in the workflow. Defaults to empty string.
        output (Path, optional): Directory path where workflow results will be
            saved. Defaults to current working directory.

    Raises:
        Exception: If the workflow execution fails due to invalid workflow file,
            missing task, or runtime errors during execution.
    """
    try:
        additional_args = ctx.args
        inputs = {
            additional_args[args_id].replace("--", ""): additional_args[args_id + 1]
            for args_id in range(0, len(additional_args), 2)
            if additional_args[args_id].startswith("--")
        }
        manager = CollectraManager()
        manager.load(workflow)
        pipeline = manager.get_pipeline()
        pipeline(task, output, force=force, **inputs)
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(error_msg(f"Failed to run task: {e}"))


if __name__ == "__main__":
    app()
