from typer import Typer, Option, Argument
from typing_extensions import Annotated
from typing import Optional, List
from rich import print
from pathlib import Path
from .models.workflow import Collectra

app = Typer()


@app.command()
def make(
    workflow: Annotated[str, Option("--workflow", "-w", help="name of the workflow")] = "default",
    file_format: Annotated[str, Option("--file-format", "-f", help="File format for the workflow, e.g., grapto, json, yaml")] = "grapto",
    output: Annotated[
        Path, Option("--output", "-o", help="Output directory for the workflow")
    ] = Path.cwd(),        
):
    """
    Create a new Collectra workflow with the specified name and file format.
    """        
    Collectra.make(name=workflow, version="1.0", output=output, file_format=file_format)

@app.command()
def add(
    workflow: Annotated[Path, Option("--workflow", "-w", help="path to workflow")],
    task: Annotated[str, Option("--task", "-t", help="task to add. It should be a valid task: task_type,task_name,engine_type,engine_name")],    
):
    """
    Add a task to the Collectra workflow
    """    
    wf = Collectra.load_workflow(workflow)
    wf.add_task(task)
    wf.save()    

@app.command()
def edit(
    workflow: Annotated[Path, Option("--workflow", "-w", help="path to workflow")],
    task: Annotated[str, Option("--task", "-t", help="task to edit")],
    param: Annotated[str, Option("--param", "-p", help="parameter to edit")],
    value: Annotated[str, Option("--value", "-v", help="new value for the parameter")],
):
    """
    Edit a specific task in the Collectra workflow
    """
    wf = Collectra.load_workflow(workflow)
    wf.edit_task(task, param, value)
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
    wf.delete_task(task)
    wf.save()            

@app.command()
def train(
    workflow: Annotated[Path, Option("--workflow", "-w", help="path to workflow")],
    task: Annotated[str, Option("--task", "-t", help="task to train")],
    input: Annotated[List[str], Argument(help="Input directory of files")],
    output: Annotated[Path, Option("--output", "-o", help="Output directory for log files")] = None,
    test: Annotated[bool, Option("--test", "-te", help="Enable test mode")] = 0,
):
    """
    Train a specific task in the Collectra workflow
    """    
    wf = Collectra.load_workflow(workflow)
    wf.train(task, input_files=input, output_path=output, test=test)    

@app.command()
def eval(
    workflow: Annotated[Path, Option("--workflow", "-w", help="path to workflow")],
    task: Annotated[str, Option("--task", "-t", help="task to evaluate")],
    input: Annotated[List[str], Argument(help="Input directory of files")],
    output: Annotated[Path, Option("--output", "-o", help="Output directory for log files")] = None,
    test: Annotated[bool, Option("--test", "-te", help="Enable test mode")] = 0,
):
    """
    Evaluate a specific task in the Collectra workflow
    """    
    wf = Collectra.load_workflow(workflow)
    wf.eval(task, input_files=input, output_path=output, test=test)

@app.command()
def view(
    workflow: Annotated[Path, Option("-w", "--workflow", help="path to workflow")],
):
    """
    View the Collectra workflow
    """    
    wf = Collectra.load_workflow(workflow)    
    for e in wf.crate.get_entities():
        print(f"[bold green]Entity:[/bold green] {e.id}") 
        print(e.properties())

if __name__ == "__main__":
    app()

