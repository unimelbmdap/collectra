import os, shutil
from typer import Typer, Option, Argument
from typing_extensions import Annotated
from typing import Optional, List
from rich import print
from pathlib import Path
from .workflow.models import Collectra
from .workflow.utils import Engine, TaskType
from .task.models import Task

app = Typer()

HELP_TEXT = f"""
    for each task, define the task type and the engine to use in the following format:\n
    <task_type>,<engine>\n
    Example: {TaskType.OBJECT_DETECT.value},{Engine.YOLO.value}\n
"""


@app.command()
def make(
    workflow: Annotated[str, Argument(help="name of the workflow")] = "default",
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
    - The tasks should be in the format: <task_type>,<engine>
    - Example: object_detect,yolo
    """    
    Collectra.make(name=workflow, version="1.0", output=output, tasks=tasks)

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
def configure(
    workflow: Path = Argument(help="path to workflow"),
    task: str = Argument(help="task to configure"),
    engine: str = Argument(help="engine to use for the task"),    
):
    wf = Collectra.load_workflow(workflow)
    for e in wf.crate.get_entities():
        if e.type == "Task" and e.id == task:
            print()
            wf.crate.add_or_update_jsonld({
                "@id": e.id,
                "@type": "Task",
                "description": e["description"],
                "task_type": e["task_type"],
                "engine": engine,                
            })   
            wf.crate.write(workflow)         
            return

@app.command()
def run(
    workflow: Path = Argument(help="path to workflow"),
):
    """
    Execute a Collectra workflow
    """
    if not workflow.exists():
        print(f"Error: The specified workflow does not exist: {workflow}")
        return
    wf = Collectra.load_workflow(workflow)
    wf.run()


def upload_grapto(
    collectra_file: Annotated[Path, Option(prompt="path to workflow")],
    grapto_files: list[Path] = None,
    grapto_file: Path = None,
):
    """
    Upload grapto files to Collectra
    """
    workflow = Collectra.load_workflow(collectra_file)
    if grapto_files:
        for file in grapto_files:
            workflow.add_file(
                file,
                dest_path=f"{file.name}",
                properties={
                    "name": file.stem,
                    "encodingFormat": "grapto",
                },
            )
        if collectra_file.is_file():
            os.remove(collectra_file)
        if collectra_file.is_dir():
            shutil.rmtree(collectra_file)
        workflow.write(collectra_file)
    elif grapto_file:
        print(f"Uploading singular grapto file: {grapto_file}")
    else:
        print("No grapto files provided for upload.")


@app.command()
def upload_graptos(
    grapto_input: Annotated[
        Path, Option(prompt="Path to the grapto files or a singular grapto file")
    ],
):
    """
    Upload grapto files to Collectra
    """
    graptos_str = str(grapto_input).strip()
    if not grapto_input.exists():
        print(f"Error: The specified path does not exist: {graptos_str}")
        return
    if grapto_input.is_dir():
        grapto_files = list(grapto_input.glob("*.grapto"))
        if graptos_str.endswith(".grapto") and not grapto_files:
            # Edge case where the directory is named like a grapto file
            upload_grapto(grapto_file=grapto_input)
        else:
            upload_grapto(grapto_files=grapto_files)


if __name__ == "__main__":
    app()
