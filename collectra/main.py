from typer import Typer, Option, Argument
from typing_extensions import Annotated
from typing import List
from rich import print
from pathlib import Path
from .pipeline import Collectra
from .utils import success_msg, error_msg
import pprint, shutil

app = Typer()


@app.command()
def make(
    pipeline: Annotated[str, Option("--workflow", "-w", help="name of the workflow")] = "default",
    version: Annotated[str, Option("--version", "-v", help="version of the workflow")] = "1.0",
    file_format: Annotated[str, Option("--file-format", "-f", help="File format for the workflow, e.g., grapto, json, yaml")] = "",    
    output: Annotated[
        Path, Option("--output", "-o", help="Output directory for the workflow")
    ] = Path.cwd(),
    as_dir: Annotated[bool, Option("--as-dir", help="Create the workflow as a directory instead of a file")] = False,
):
    """
    Create a new Collectra workflow with the specified name and file format.
    """            
    Collectra.make(
        name=pipeline,
        version=version,
        output=output,
        file_format=file_format,
        as_dir=as_dir,
    )
    print(success_msg(f"Workflow '{pipeline}' created at {output}"))

@app.command()
def render(
    workflow: Annotated[Path, Option("-w", "--workflow", help="path to workflow")],
    output: Annotated[Path, Option("-o", "--output", help="path to output file")] = Path.cwd() / "workflow.png",
):
    """
    Render the Collectra workflow to a file
    """
    try:
        print(f"Rendering workflow [green]{workflow}[/green] to {output}")
        pipeline = Collectra.load(workflow)        
        pprint.pprint(pipeline.metadata())
        pprint.pprint(f"List of tasks: {pipeline.tasks}")
        print(success_msg(f"Finished rendering from {pipeline}"))                 
    except Exception as e:
        print(error_msg(f"{e}"))    
    finally:
        tmp_path = Path("tmp")
        if tmp_path.exists():
            shutil.rmtree(tmp_path, ignore_errors=True)  # Clean up temporary files   

@app.command()
def add(
    workflow: Annotated[Path, Option("--workflow", "-w", help="path to workflow")],
    task: Annotated[str, Option("--task", "-t", help="task to add. It should be a valid task: task_type,task_name,engine_type,engine_name")],
    task_input: Annotated[str, Option("--input", "-i", help="valid input name for the task")] = "",
    task_output: Annotated[List[str], Option("--output", "-o", help="valid output name for the task")] = [],
):
    """
    Add a task to the Collectra workflowj
    """    
    try:
        pipeline = Collectra.load(workflow)            
        if pipeline.add(task):
            pipeline.save()         
    except Exception as e:
        print(error_msg(f"Failed to add task: {e}"))

@app.command()
def train(
    pipeline: Annotated[Path, Option("--workflow", "-w", help="path to workflow")],
    task: Annotated[str, Option("--task", "-t", help="task to train")],
    input: Annotated[List[str], Argument(help="Input directory of files")],
    output: Annotated[Path, Option("--output", "-o", help="Output directory for log files")] = Path.cwd() / "tmp",
    epochs: Annotated[int, Option("--epochs", help="Number of epochs for training")] = 1,
    imgsz: Annotated[int, Option("--imgsz", help="Image size for training")] = 640,
    lr0: Annotated[float, Option("--lr0", help="Learning rate for training")] = 0.01,
):
    """
    Train a specific task in the Collectra workflow
    """    
    config = {
        "input": input,
        "output": output,
        "epochs": epochs,
        "imgsz": imgsz,
        "lr0": lr0,
    }
    Collectra.load(pipeline).train(task_id=task, config=config)

@app.command()
def run(
    workflow: Annotated[Path, Option("--workflow", "-w", help="path to workflow")],
    task: Annotated[str, Option("--task", "-t", help="task to run")],
    images: Annotated[List[str], Argument(help="Input directory of files")],
    as_dir: Annotated[bool, Option("--as-dir", help="Run the task as a directory instead of a file")] = False,
):
    config = {
        "images": images,
        "as_dir": as_dir,
    }
    Collectra.load(workflow).run(task_id=task, config=config)
    
    
if __name__ == "__main__":
    app()

