import yaml, logging
from typer import Typer, Option, Argument
from typing_extensions import Annotated
from typing import List
from rich import print
from pathlib import Path
from .pipeline import Collectra, CollectraManager
from .utils import success_msg, error_msg

app = Typer()


@app.command()
def make(
    dir: Annotated[str, Option("--workflow", "-w", help="name of the workflow")],
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
    """Create a new Collectra pipeline.

    Args:
        name (str): Name of the pipeline.
        version (str): Version of the pipeline.
        file_format (str): File format for the pipeline, e.g., grapto, hespi, etc.
        out_dir (Path): Output directory for the pipeline.
        as_dir (bool): Create the pipeline as a directory instead of a file.
    """
    input_directory = Path(dir)
    name = input_directory.name
    out_dir = input_directory.parent
    pipeline: Collectra = Collectra.make(
        name=name,
        version=version,
        file_format=file_format,
        out_dir=out_dir,
        as_dir=as_dir,
    )
    pipeline.save()
    print(success_msg(f"Workflow '{pipeline}' created at {name}"))


@app.command()
def render(
    workflow: Annotated[Path, Option("-w", "--workflow", help="path to workflow")],
    output: Annotated[
        Path, Option("-o", "--output", help="path to output file")
    ] = Path.cwd()
    / "workflow.png",
):
    """
    Render the Collectra workflow to a file
    """
    try:
        print(f"Rendering workflow [green]{workflow}[/green] to {output}")
        pipeline = CollectraManager.load(workflow)
        if pipeline.tasks:
            print(f"\n[yellow2]Tasks in the pipeline:[/yellow2]\n")
        for task in pipeline.tasks:
            print(yaml.dump(task.metadata(), default_flow_style=False, sort_keys=False))
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
        List[str], Option("--input", "-i", help="valid input name for the task")
    ] = [],
    task_output: Annotated[
        List[str], Option(help="valid output name for the task")
    ] = [],
):
    """
    Add a task to the Collectra workflowj
    """
    try:
        CollectraManager.load(workflow).add(task, task_input, task_output).save()
    except Exception as e:
        print(error_msg(f"Failed to add task: {e}"))


@app.command()
def train(
    pipeline: Annotated[Path, Option("--workflow", "-w", help="path to workflow")],
    task: Annotated[str, Option("--task", "-t", help="task to train")],
    input_files: Annotated[List[str], Argument(help="Input directory of files")],
    output_log: Annotated[
        Path, Option("--output", "-o", help="Output directory for log files")
    ] = Path("logs"),
    epochs: Annotated[
        int, Option("--epochs", help="Number of epochs for training")
    ] = 1,
    imgsz: Annotated[int, Option("--imgsz", help="Image size for training")] = 640,
    lr0: Annotated[float, Option("--lr0", help="Learning rate for training")] = 0.01,
):
    """
    Train a specific task in the Collectra workflow
    """
    config = {
        "input_files": input_files,
        "output_log": output_log,
        "epochs": epochs,
        "imgsz": imgsz,
        "lr0": lr0,
    }
    CollectraManager.load(pipeline).train(task_id=task, config=config)


@app.command()
def run(
    workflow: Annotated[Path, Option("--workflow", "-w", help="path to workflow")],
    task: Annotated[str, Option("--task", "-t", help="task to run")],
    images: Annotated[List[str], Argument(help="Input directory of files")],
    as_dir: Annotated[
        bool, Option("--as-dir", help="Run the task as a directory instead of a file")
    ] = False,
):
    config = {"images": images, "as_dir": as_dir}
    CollectraManager.load(workflow).run(task_id=task, config=config)


if __name__ == "__main__":
    app()
