import os, shutil
from typer import Typer, Option, Argument
from typing_extensions import Annotated
from typing import Optional, List
from rich import print
from pathlib import Path
from .models.workflow import Collectra
from .models.utils import Engine, TaskType
from .models.task import TaskEntity

app = Typer()

HELP_TEXT = f"""
    for each task, define the task type and the engine to use in the following format:\n
    <task_type>,<engine>\n
    Example: {TaskType.OBJECT_DETECT.value},{Engine.YOLO.value}\n
"""


@app.command()
def make(
    workflow: Annotated[str, Argument(help="name of the workflow")] = "default",
    file_format: Annotated[str, Option("--file-format", "-f", help="File format for the workflow, e.g., grapto, json, yaml")] = "grapto",
    tasks: Annotated[
        Optional[List[str]],
        Option("--task", "-t", help=HELP_TEXT, case_sensitive=False),
    ] = [],
    output: Annotated[
        Path, Option("--output", "-o", help="Output directory for the workflow")
    ] = Path.cwd(),
):
    """
    Build a collectra workflow
    - Checks if the workflow already exists in the output directory
    - If it exists, skips the initialisation
    - If it does not exist, creates a new workflow with the specified tasks
    - If the workflow is 'default', it overrides the default workflow folder
    - If the output directory does not exist, it creates it
    - If the tasks are not provided, it creates an empty workflow
    - If the tasks are provided, it creates a workflow with the specified tasks
    - The tasks should be in the format: <task_type>,<engine_type>,<engine>
    - <engine> should be a valid Path to an engine file
    - Example: object_detect,yolo,yolo11n.pt
    """    
    Collectra.make(name=workflow, version="1.0", output=output, tasks=tasks, file_format=file_format)

@app.command()
def add(
    workflow: Annotated[Path, Argument(help="path to workflow")],
    tasks: Annotated[
        Optional[List[str]],
        Option("--task", "-t", help=HELP_TEXT, case_sensitive=False),
    ] = [],
):
    """
    Add a task to the Collectra workflow
    """    
    wf = Collectra.load_workflow(workflow)
    wf.add_tasks(tasks)    
    wf.save()    

@app.command()
def delete(
    workflow: Annotated[Path, Argument(help="path to workflow")],
    task: Annotated[str, Option("--task", "-t", help="task to remove")],
):
    """
    Remove a task from the Collectra workflow
    """    
    wf = Collectra.load_workflow(workflow)
    try:
        wf.delete(task)
        wf.save()        
    except ValueError as e:
        print(f"[red]Error removing task[/red]: {e}")

@app.command()
def train(
    workflow: Annotated[Path, Argument(help="path to workflow")],
    task: Annotated[str, Option("--task", "-t", help="task to train")],
    input: Annotated[Path, Option("--input", "-i", help="Input directory of files")] = Path.cwd() / "data" / "images",
    output: Annotated[Path, Option("--output", "-o", help="Output directory for log files")] = Path.cwd() / "output",    
):
    """
    Train a specific task in the Collectra workflow
    """    
    wf = Collectra.load_workflow(workflow)
    try:
        wf.train(task, input=input, output=output)        
    except ValueError as e:
        print(f"[red]Error training task[/red]: {e}")    

@app.command()
def run(
    workflow: Path = Argument(help="path to workflow"),
    input: Annotated[Path, Option("--input", "-i", help="Input directory of files")] = Path.cwd() / "data"/ "images",
    output: Annotated[Path, Option("--output", "-o", help="Output directory for processed files")] = Path.cwd() / "data" / "output",
):
    """
    Execute a Collectra workflow
    """    
    wf = Collectra.load_workflow(workflow)
    wf.run(input=input, output=output)

@app.command()
def view(
    workflow: Annotated[Path, Argument(help="path to workflow")],
):
    """
    View the Collectra workflow
    """    
    wf = Collectra.load_workflow(workflow)    
    print(wf)


