import os, shutil
from typer import Typer, Option
from .workflow.models import Collectra, Grapto
from typing_extensions import Annotated
from rich import print
from pathlib import Path
from rocrate.rocrate import ROCrate

app = Typer()

@app.command()
def init(
    workflow_name: Annotated[str, Option(prompt="Workflow name")],
    output: Annotated[Path, Option(prompt="Output directory for the workflow")],
):
    """
    Initiate a collectra workflow
    """        
    Collectra.build(
        name=workflow_name, 
        version="1.0",
        output=output
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
        grapto_files: list[Path] = None,
        grapto_file: Path = None,
):
    """
    Upload grapto files to Collectra
    """
    collectra_file = Path("sample_data/flow.collectra")
    workflow = ROCrate(collectra_file)    
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

