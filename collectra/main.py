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

import logging
import os
import shutil
import sys
import traceback
from pathlib import Path

import typer
import yaml
from rich.console import Console
from typing_extensions import Annotated

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout)

app = typer.Typer()


def resolve_workflow_path(workflow: Path):
    from collectra import Collectra

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
        pipeline.render(dest, str(dest), True)
    except Exception as e:
        traceback.print_exc()


@app.command()
def train(
    workflow: Annotated[
        Path, typer.Option("--workflow", "-w", help="path to workflow")
    ],
    task: Annotated[str, typer.Option("--task", "-t", help="task to train")],
    input: Annotated[list[str], typer.Argument(help="Input directory of files")],
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
        from datetime import datetime

        pipeline = resolve_workflow_path(workflow)

        project = f"{workflow.name}-{task}"

        log = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        config = {
            "input": input,
            "project": project,
            "log": log,
            "base_folder": Path.cwd(),
        }

        pipeline.train(task, **config)
        pipeline.save(task)

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
    inputs: Annotated[list[Path], typer.Argument(help="Input files for the workflow")],
    task: Annotated[str, typer.Option("--task", "-t", help="task to run")] = "",
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="output directory")
    ] = None,
    single: Annotated[
        bool, typer.Option("--single", help="Runs only the designated task")
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose", "-v", help="Enables verbose output during workflow execution"
        ),
    ] = False,
    usage: Annotated[
        bool,
        typer.Option("--usage", help="Enables token usage tracking during LLM tasks"),
    ] = False,
    render: Annotated[
        bool,
        typer.Option(
            "--render", help="Enables rendering of the workflow state during execution"
        ),
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
    console = Console()
    try:
        data: dict[str, str | bool | list[str] | Path | list[Path]] = dict()
        data["single"] = single
        data["usage"] = usage
        data["render"] = render
        if output:
            data["output"] = str(output)
        data["files"] = list()
        pipeline = resolve_workflow_path(workflow)
        for input_path in inputs:
            input_path = Path(input_path)
            if input_path.is_file() or (
                input_path.is_dir() and input_path.suffix.lower() == f".{pipeline.ext}"
            ):
                data["files"].append(input_path)
            elif input_path.is_dir():
                data["files"] += [
                    file
                    for file in input_path.rglob("*")
                    if file.is_file()
                    or (file.is_dir() and file.suffix.lower() == pipeline.ext)
                ]
        pipeline(task, **data)
        if verbose:
            console.print(pipeline.log)
    except Exception as e:
        console.print(traceback.format_exc())
        console.print(e)


@app.command()
def evaluate(
    workflow: Annotated[
        Path, typer.Option("--workflow", "-w", help="path to workflow")
    ],
    predicted: Annotated[
        Path, typer.Argument(help="folder of collectra files for evaluation")
    ],
    gold: Annotated[
        Path, typer.Option("--gold", "-g", help="folder of gold standard files")
    ],
):
    """Evaluate predicted results against gold standard files.

    Compares predicted output files against ground truth files and computes
    precision, recall, F1-score, and other metrics. The evaluation is performed
    across all labels and aggregated into a comprehensive report.

    Args:
        workflow: Path to the workflow configuration directory.
        predicted: Folder containing predicted .{ext} files to evaluate.
        gold: Folder containing gold standard .{ext} files for comparison.
    """
    from rich.columns import Columns

    console = Console()
    try:
        from collectra import Evaluator

        pipeline = resolve_workflow_path(workflow)
        evaluator = Evaluator(
            predicted,
            gold,
            pipeline.ext,
        )
        report = evaluator.evaluate()
        if report.aggregate_tables:
            console.print(Columns(report.aggregate_tables))
        for table in report.tables:
            console.print(table)
    except Exception as e:
        console.print(traceback.format_exc())


@app.command()
def view(
    file: Path = typer.Option("--file", help="The collectra result file to be viewed")
):
    try:
        from collectra import Editor

        editor = Editor(file)
        editor.view()
    except Exception as e:
        traceback.print_exc()


@app.command()
def convert(
    config: Annotated[str, typer.Argument(help="path to conversion config file")],
):
    """Convert files based on the provided configuration.

    Args:
        config (Path): Path to the configuration file for conversion.

    Raises:
        Exception: If the conversion process fails due to invalid configuration
            or runtime errors during execution.
    """
    try:
        from collectra.utils import convert_files

        convert_files(config)
    except Exception as e:
        traceback.print_exc()


@app.command()
def ensemble(
    workflow: Annotated[
        Path, typer.Option("--workflow", "-w", help="path to workflow")
    ],
    folders: Annotated[
        list[Path], typer.Argument(help="Input folders containing collectra files")
    ] = [],
    ensemble_folder: Annotated[
        Path,
        typer.Option(
            "--ensemble-folder",
            "-e",
            help="Folder containing folders of collectra files to ensemble",
        ),
    ] = Path("."),
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Output folder for ensemble results")
    ] = Path("ensemble_results"),
):
    console = Console()
    try:
        if not folders and ensemble_folder.exists():
            for item in ensemble_folder.iterdir():
                if item.is_dir():
                    folders.append(item)
        pipeline = resolve_workflow_path(workflow)
        pipeline.ensemble(folders, output)
    except Exception as e:
        console.print(traceback.format_exc())


@app.command()
def analyse(
    folder: Annotated[
        Path, typer.Argument(help="Input folder containing collectra files")
    ],
    extension: Annotated[
        str, typer.Option("--ext", "-e", help="File extension of collectra files")
    ],
):
    console = Console()
    try:
        feature_count = dict()
        for item in folder.rglob(f"*.{extension}"):
            data_yaml = item / "results.yaml"
            if not data_yaml.exists():
                continue
            with open(data_yaml, "r") as f:
                data = yaml.safe_load(f)
            data.pop("collectra_results_metadata", None)
            for key, value in data.items():
                if not isinstance(value, list):
                    value = [value]
                feature_count.setdefault(key, 0)
                feature_count[key] += len(value)

        from rich.table import Table

        table = Table(title="Dataset Feature Counts")
        table.add_column("Feature", style="cyan", no_wrap=True)
        table.add_column("Count", style="magenta")
        for feature, count in feature_count.items():
            table.add_row(feature, str(count))
        console.print(table)
    except Exception as e:
        traceback.print_exc()
        console.print(e)


if __name__ == "__main__":
    app()
