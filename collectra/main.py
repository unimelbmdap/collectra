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

import os, shutil, logging, sys, typer, yaml, traceback, webview

from datetime import datetime
from pathlib import Path
from typing_extensions import Annotated

from collectra import Collectra, Viewer

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout)

app = typer.Typer()


def resolve_workflow_path(workflow: Path) -> Collectra:
    with open(workflow / "pipeline.yaml", "r") as f:
        metadata = yaml.safe_load(f)
    initials: dict = metadata.pop("collectra_pipeline_metadata")
    name = initials["name"]
    ext = initials["ext"]
    version = initials["version"]
    pipeline = Collectra(name, ext, version, path=str(workflow), **metadata)
    return pipeline


@app.command()
def render(
    workflow: Annotated[
        Path, typer.Option(..., "-w", "--workflow", help="path to workflow")
    ],
    dest: Annotated[Path, typer.Option(..., "-d", "--dest", help="output file")],
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
        pipeline = resolve_workflow_path(workflow)
        dest.parent.mkdir(parents=True, exist_ok=True)
        pipeline.render(dest)
    except Exception as e:
        traceback.print_exc()


@app.command()
def train(
    workflow: Annotated[
        Path, typer.Option("--workflow", "-w", help="path to workflow")
    ],
    task: Annotated[str, typer.Option("--task", "-t", help="task to train")],
    input_files: Annotated[list[str], typer.Argument(help="Input directory of files")],
    keep_log: Annotated[
        bool, typer.Option("--keep-log", help="Keep previous log files")
    ] = True,
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
        log = Path.cwd() / f"{task}_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        config = {"input": input_files, "log_dir": log}
        pipeline = resolve_workflow_path(workflow)
        pipeline.train(task, **config)
        pipeline.save()
        if not keep_log:
            shutil.rmtree(log, ignore_errors=True)
            log_cache = Path(f"{log}.cache")
            if log_cache.exists():
                os.remove(log_cache)
    except Exception as e:
        traceback.print_exc()


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
@app.command()
def run(
    workflow: Annotated[
        Path, typer.Option("--workflow", "-w", help="path to workflow")
    ],
    ctx: typer.Context,
    task: Annotated[str, typer.Option("--task", "-t", help="task to run")] = "",
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="output directory")
    ] = None,
    single: Annotated[
        bool, typer.Option("--single", help="Runs only the designated task")
    ] = False,
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
        data: dict[str, str | bool] = {
            additional_args[args_id].replace("--", ""): additional_args[args_id + 1]
            for args_id in range(0, len(additional_args), 2)
            if additional_args[args_id].startswith("--")
        }
        data["single"] = single
        if output:
            data["output"] = str(output)
        pipeline = resolve_workflow_path(workflow)
        pipeline(task, **data)
    except Exception as e:
        traceback.print_exc()


@app.command()
def edit(
    file: Annotated[Path | None, typer.Option("--file", help="The collectra result file to be edited")] = None,
    ssl: Annotated[bool, typer.Option("--ssl", help="Enable SSL for the webview")] = False,
    debug: Annotated[bool, typer.Option("--debug", help="Enable debug mode for the webview")] = False,
):
    try:
        if not file:            
            viewer = Viewer()
            webview.create_window("Collectra viewer", url= str(Path.cwd() / "collectra/editor/templates/index.html"), js_api=viewer, min_size=(600,450))            
            webview.start(ssl=ssl, debug=debug)
        else:   
            editor = Viewer(file)         
            editor.edit()
    except Exception as e:
        traceback.print_exc()

