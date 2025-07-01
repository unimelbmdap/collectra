import os, shutil
from typer import Typer, Option, Argument
from typing_extensions import Annotated
from typing import Optional, List
from rich import print
from pathlib import Path
from .workflow.models import Collectra, Grapto
from .workflow.utils import Engine, TaskType

app = Typer()    

HELP_TEXT = f"""
    for each task, define the task type and the engine to use in the following format:\n
    <task_type>,<engine>\n
    Example: {TaskType.OBJECT_DETECT.value},{Engine.YOLO.value}\n
"""

@app.command()
def create(
    workflow: Annotated[str, Argument(help="name of the workflow")] = "default",
    tasks: Annotated[Optional[List[str]], Option("--task", "-t", help=HELP_TEXT, case_sensitive=False)] = [],           
    output: Annotated[Path, Option("--output", "-o", help="Output directory for the workflow")] = Path.cwd(),        
):
    """
    Build a collectra workflow
    """
    workflow_p = Path(f"{output}/{workflow}.collectra")
    if workflow_p.exists() and workflow != "default":        
        print(f"Workflow file already exists: {workflow} - skipping initialisation. To edit use `collectra edit` command.")        
    else:
        if workflow == "default":
            # Override the default workflow folder
            shutil.rmtree(workflow_p, ignore_errors=True)
        os.makedirs(workflow_p.parent, exist_ok=True)
        Collectra.create(
            name=workflow, 
            version="1.0",
            output=output,
            tasks=tasks
        )               

@app.command()
def generate_grapto(
    grapto_name: Annotated[str, Option(prompt="Grapto name")],
    output: Annotated[Path, Option(prompt="Output directory for the grapto")],
):
    """
    Generate a grapto for Collectra
    """
    Grapto.generate_grapto(
        name=grapto_name, 
        version="1.0",
        output=output
    )

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
                }
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
    grapto_input: Annotated[Path, Option(prompt="Path to the grapto files or a singular grapto file")],    
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
        if graptos_str.endswith('.grapto') and not grapto_files:
            # Edge case where the directory is named like a grapto file                                
            upload_grapto(grapto_file=grapto_input)                    
        else:                
            upload_grapto(grapto_files=grapto_files)

@app.command()
def set():
    """
    Set a configuration for Collectra
    """
    print("Setting configuration...")       

if __name__ == "__main__":
    app()

