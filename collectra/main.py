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
    workflow_p = Path(f"{output}/{workflow}.collectra")
    if workflow_p.exists() and workflow != "default":
        print(f"Workflow file already exists: {workflow} - skipping initialisation. To edit use `collectra edit` command.")
        return    
    print(f"Creating workflow: {workflow} in {output}")
    if workflow == "default":
        shutil.rmtree(workflow_p, ignore_errors=True)
    os.makedirs(workflow_p.parent, exist_ok=True)
    Collectra.make(name=workflow, version="1.0", output=output, tasks=tasks)

@app.command()
def add(
    workflow: Annotated[Path, Argument(help="path to workflow")],
    task: Annotated[
        str, Argument(help="task to add in the format <task_type>,<engine>")
    ],
):
    """
    Add a task to the Collectra workflow
    """
    if not workflow.exists():
        print(f"Error: The specified workflow does not exist: {workflow}")
        return
    wf = Collectra.load_workflow(workflow)
    task_type, engine = task.split(",")
    wf.add_task(task_type=task_type, engine=engine)
    wf.write(workflow)

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


@app.command()
def configure(
    workflow: Path = Argument(help="path to workflow"),
    task: str = Argument(help="task to configure"),
    engine: str = Argument(help="engine to use for the task"),
    input: Path = Argument(help="input path for the task"),
    output: Annotated[Path, Argument(help="output path for the task")] = "",
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
                "input": str(input),
                "output": str(output),
            })   
            wf.crate.write(workflow)         
            return


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
