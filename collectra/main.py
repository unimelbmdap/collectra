from datetime import datetime
from pathlib import Path
from rich import print
import os, shutil
from typer import Typer, Option, Argument
from typing_extensions import Annotated
from collectra.pipeline import Collectra, CollectraManager
from collectra.utils import success_msg, error_msg

app = Typer()


@app.command()
def make(
    workflow_path: Annotated[
        str, Option("--workflow", "-w", help="name of the workflow")
    ],
    version: Annotated[str, Option("--version", "-v", help="version of the workflow")],
    file_format: Annotated[
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

    Args:
        workflow_path (str): Name of the workflow.
        version (str): Version of the workflow.
        file_format (str): File format for the workflow, e.g., grapto, hespi, etc.
        out_dir (Path): Output directory for the workflow.
        as_dir (bool): Create the workflow as a directory instead of a file.
    
    Raises:
        Exception: If the workflow cannot be created
    """
    try:
        out_dir = Path(workflow_path)
        name = out_dir.name
        workflow: Collectra = Collectra.make(
            name=name,
            version=version,
            file_format=file_format,
            out_dir=out_dir,
            as_dir=as_dir,
        )
        workflow.save()
        print(success_msg(f"Workflow '{workflow}' created at {name}"))
    except Exception as e:
        print(error_msg(f"Failed to create workflow: {e}"))


@app.command()
def render(
    workflow: Annotated[Path, Option("-w", "--workflow", help="path to workflow")],
):
    """Render the Collectra workflow to a file

    Args:
        workflow (Path): Path to the workflow file
    
    Raises:
        Exception: If the workflow cannot be rendered
    """
    try:
        CollectraManager.load(workflow).render()
    except Exception as e:
        print(error_msg(f"{e}"))


@app.command()
def add(
    workflow: Annotated[Path, Option("--workflow", "-w", help="path to workflow")],
    task: Annotated[
        str,
        Option(
            "--task",
            "-t",
            help="task to add. It should be a valid task: task_type,task_name,engine_type,engine_name",
        ),
    ],
    task_input: Annotated[
        list[str], Option("--input", "-i", help="valid input name for the task")
    ] = [],
    task_output: Annotated[
        list[str], Option(help="valid output name for the task")
    ] = [],
):
    """Add a task to the Collectra workflow

    Args:
        workflow (Path): Path to the workflow file
        task (str): Task to add. It should be a valid task: task_name,task_type,engine_name
        task_input (list[str]): Valid inputs 
        task_output (list[str]): Valid outputs 
    
    Raises:
        Exception: If the task cannot be added

    """
    try:
        CollectraManager.load(workflow).add(task, task_input, task_output).save()
    except Exception as e:
        print(error_msg(f"Failed to add task: {e}"))


@app.command()
def train(
    workflow: Annotated[Path, Option("--workflow", "-w", help="path to workflow")],
    task: Annotated[str, Option("--task", "-t", help="task to train")],
    input_files: Annotated[list[str], Argument(help="Input directory of files")],    
    keep_log: Annotated[bool, Option("--keep-log", help="Keep previous log files")] = False,
):
    """Train a specific task in the Collectra workflow

    Args:
        workflow (Path): Path to the workflow file
        task (str): Task to train
        input_files (list[str]): Input directory of files
        output_log (Path): Output directory for log files
    
    Raises:
        Exception: If the task cannot be trained
    """
    try:
        output_log = f"{task}_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        config = {"input_files": input_files, "output_log": output_log}
        CollectraManager.load(workflow).train(task_name=task, config=config)
        if not keep_log:
            shutil.rmtree(output_log, ignore_errors=True)            
            log_cache = Path(f"{output_log}.cache")
            if log_cache.exists():
                os.remove(log_cache)               
    except Exception as e:
        print(error_msg(f"Failed to train task: {e}"))


@app.command()
def run(
    workflow: Annotated[Path, Option("--workflow", "-w", help="path to workflow")],
    task: Annotated[str, Option("--task", "-t", help="task to run")],
    images: Annotated[list[str], Argument(help="Input directory of files")],
    as_dir: Annotated[
        bool, Option("--as-dir", help="Run the task as a directory instead of a file")
    ] = False,
):
    try:
        config = {"images": images, "as_dir": as_dir}
        CollectraManager.load(workflow).run(task_name=task, config=config)
    except Exception as e:
        print(error_msg(f"Failed to run task: {e}"))


if __name__ == "__main__":
    app()
