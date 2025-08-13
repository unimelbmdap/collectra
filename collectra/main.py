from typer import Typer, Option, Argument
from typing_extensions import Annotated
from typing import Optional, List
from rich import print
from pathlib import Path
# from .models.workflow import Collectra
from .pipeline import Collectra
from .parsing import CollectraWorkflow
import zipfile, pprint

app = Typer()


@app.command()
def make(
    pipeline: Annotated[str, Option("--workflow", "-w", help="name of the workflow")] = "default",
    version: Annotated[str, Option("--version", "-v", help="version of the workflow")] = "1.0",
    file_format: Annotated[str, Option("--file-format", "-f", help="File format for the workflow, e.g., grapto, json, yaml")] = None,    
    output: Annotated[
        Path, Option("--output", "-o", help="Output directory for the workflow")
    ] = Path.cwd(),
    as_dir: Annotated[bool, Option("--as-dir", "-d", help="Create the workflow as a directory instead of a file")] = False,
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
    print(f"[green]Success[/green] Workflow '{pipeline}' created at {output}")

@app.command()
def render(
    workflow: Annotated[Path, Option("-w", "--workflow", help="path to workflow")],
    output: Annotated[Path, Option("-o", "--output", help="path to output file")] = Path.cwd() / "workflow.png",
):
    """
    Render the Collectra workflow to a file
    """
    pipeline = Collectra.load(workflow)
    print(f"Rendering workflow [green]{workflow}[/green] to {output}")
    pprint.pprint(pipeline.get_metadata())
    pprint.pprint(f"List of tasks: {pipeline.tasks}")


@app.command()
def add(
    workflow: Annotated[Path, Option("--workflow", "-w", help="path to workflow")],
    task: Annotated[str, Option("--task", "-t", help="task to add. It should be a valid task: task_type,task_name,engine_type,engine_name")],            
    output: Annotated[Path, Option("--output", "-o", help="Output directory for the workflow")] = Path.cwd(),
    as_dir: Annotated[bool, Option("--as-dir", "-d", help="Create the workflow as a directory instead of a file")] = False,
):
    """
    Add a task to the Collectra workflow
    """    
    pipeline = Collectra.load(workflow)    
    add_outcome = pipeline.add(task)    
    if add_outcome:
        pipeline.save(output=output, as_dir=as_dir)    

# @app.command()
# def edit(
#     workflow: Annotated[Path, Option("--workflow", "-w", help="path to workflow")],
#     task: Annotated[str, Option("--task", "-t", help="task to edit")],
#     param: Annotated[str, Option("--param", "-p", help="parameter to edit")],
#     value: Annotated[str, Option("--value", "-v", help="new value for the parameter")],
# ):
#     """
#     Edit a specific task in the Collectra workflow
#     """
#     wf = Collectra.load_workflow(workflow)
#     wf.edit_task(task, param, value)
#     wf.save()

# @app.command()
# def delete(
#     workflow: Annotated[Path, Argument(help="path to workflow")],
#     task: Annotated[str, Option("--task", "-t", help="task to remove")],
# ):
#     """
#     Remove a task from the Collectra workflow
#     """    
#     wf = Collectra.load_workflow(workflow)    
#     wf.delete_task(task)
#     wf.save()            

@app.command()
def train(
    pipeline: Annotated[Path, Option("--workflow", "-w", help="path to workflow")],
    task: Annotated[str, Option("--task", "-t", help="task to train")],
    input: Annotated[List[str], Argument(help="Input directory of files")],
    output: Annotated[Path, Option("--output", "-o", help="Output directory for log files")] = Path.cwd() / "tmp",
    epochs: Annotated[int, Option("--epochs", help="Number of epochs for training")] = 1,
    imgsz: Annotated[int, Option("--imgsz", help="Image size for training")] = 640,
):
    """
    Train a specific task in the Collectra workflow
    """    
    config = {
        "input": input,
        "output": output,
        "epochs": epochs,
        "imgsz": imgsz,        
    }
    Collectra.load(pipeline).train(task_id=task, config=config)

# @app.command()
# def eval(
#     workflow: Annotated[Path, Option("--workflow", "-w", help="path to workflow")],
#     task: Annotated[str, Option("--task", "-t", help="task to evaluate")],
#     input: Annotated[List[str], Argument(help="Input directory of files")],
#     output: Annotated[Path, Option("--output", "-o", help="Output directory for log files")] = None,
#     test: Annotated[bool, Option("--test", "-te", help="Enable test mode")] = 0,
# ):
#     """
#     Evaluate a specific task in the Collectra workflow
#     """    
#     wf = Collectra.load_workflow(workflow)
#     wf.eval(task, input_files=input, output_path=output, test=test)
    
# @app.command()
# def cluster(
#     workflow: Annotated[Path, Option("--workflow", "-w", help="path to workflow")],
#     item: Annotated[str, Option("--item", "-i", help="item to cluster")],
#     input: Annotated[List[str], Argument(help="Input directory of files")],
#     output: Annotated[Path, Option("--output", "-o", help="Output directory for log files")] = None,
# ):
#     """
#     Cluster items in the Collectra workflow
#     """
#     wf = Collectra.load_workflow(workflow)
#     wf.cluster(item, input_files=input, output_path=output)

# @app.command()
# def view(
#     workflow: Annotated[Path, Option("-w", "--workflow", help="path to workflow")],
# ):
#     """
#     View the Collectra workflow
#     """    
#     wf = Collectra.load_workflow(workflow)    
#     for e in wf.crate.get_entities():
#         print(f"[bold green]Entity:[/bold green] {e.id}") 
#         print(e.properties())


# @app.command()
# def render(
#     workflow: Annotated[Path, Argument(help="path to workflow")],
#     output: Annotated[Path, Argument(help="path to output file")],
# ):
#     """
#     Render the Collectra workflow to a file
#     """
#     workflow = CollectraWorkflow(workflow)
#     workflow.render(output)

#     print(f"[green]Workflow rendered to {output}[/green]")

    
if __name__ == "__main__":
    app()

