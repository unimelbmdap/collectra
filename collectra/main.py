"""Main command-line interface for the Collectra workflow management system.

This module provides the CLI commands for creating, rendering, training, and running
Collectra workflows. It serves as the entry point for the application and handles
user interactions through the Typer framework.

The module supports the following operations:
    - Creating new workflow configurations
    - Rendering workflow diagrams
    - Training machine learning tasks within workflows
    - Executing complete workflows or specific tasks

Example:
    $ collectra make --workflow my_workflow --version 1.0
    $ collectra run --workflow pipeline.yaml --task detection
"""

import os, shutil, logging, sys, typer, yaml, traceback, tempfile, re

import numpy as np

from datetime import datetime
from pathlib import Path
from typing_extensions import Annotated
from typing import List

from collectra import Collectra, Editor

from collectra.utils import change_dir, load_class_from_string


# from imcluster.io import ImclusterIO
# from imcluster.features import build_features
# from imcluster.pca import fit_pca
# from imcluster.cluster import cluster
# from imcluster.plotting import plot
# from imcluster.html import write_html


logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout)

app = typer.Typer()

# def clusterer(
#         inputs:list[Path],
#         output_df:Path,
#         output_html:Path = None,
#         model:str = typer.Option("vgg19", help="The name of the torchvision model to use (see https://pytorch.org/vision/stable/models.html#)."),
#         max_images:int = None,
#         algorithm:str = "SPECTRAL",
#         n_clusters:int = 20,
#         batch_size:int = 1,
#         thumbnail_width:int = 256,
#         thumbnail_height:int = 256,
#         force:bool = False,
#         force_features:bool = False,
#         force_pca:bool = False,
#         force_cluster:bool = False,
#         force_thumbnails:bool = False,
#     ):    
#         imcluster_io = ImclusterIO(inputs, output_df, max_images=max_images)
#         feature_vectors = build_features(
#             imcluster_io, model_name=model, force=force or force_features
#         )

#         fit_pca(imcluster_io, feature_vectors, force=force or force_features or force_pca)

#         cluster(
#             imcluster_io,
#             feature_vectors,
#             algorithm=algorithm,
#             n_clusters=n_clusters,
#             force=force or force_features or force_cluster,
#         )
#         # save_clusters(
#         # imcluster_io=imcluster_io, output_dir=output_path, algorithm=algorithm
#         # )
#         plot(imcluster_io, output_html, thumbnail_height=thumbnail_height, thumbnail_width=thumbnail_width, force_thumbnails=force_thumbnails)
#         write_html(imcluster_io)
#         return imcluster_io.images.copy(), feature_vectors

def resolve_workflow_path(workflow: Path) -> Collectra:
    with open(workflow / "pipeline.yaml", "r") as f:
        metadata = yaml.safe_load(f)
    initials: dict = metadata.pop("collectra_pipeline_metadata")
    name = initials["name"]
    ext = initials["ext"]
    version = initials["version"]
    pipeline = Collectra(name, ext, version, path=str(workflow), **metadata)
    return pipeline


@app.command()
def render(
    workflow: Annotated[
        Path, typer.Option(..., "-w", "--workflow", help="path to workflow")
    ],
    dest: Annotated[Path, typer.Option(..., "-d", "--dest", help="output file")],
):
    """Render the Collectra workflow as a visual diagram.

    Generates an SVG visualization of the workflow showing tasks and their
    dependencies. The diagram can be rendered in raw format (showing the
    underlying graph structure) or formatted view (showing inputs/outputs).

    Args:
        workflow (Path): Path to the workflow configuration file to render.
        raw (bool, optional): Whether to render the raw pipeline graph structure
            instead of the formatted view. Defaults to False.

    Raises:
        Exception: If the workflow file cannot be loaded or rendered due to
            invalid format or missing dependencies.
    """
    try:
        pipeline = resolve_workflow_path(workflow)
        dest.parent.mkdir(parents=True, exist_ok=True)
        pipeline.render(dest)
    except Exception as e:
        traceback.print_exc()


@app.command()
def train(
    workflow: Annotated[
        Path, typer.Option("--workflow", "-w", help="path to workflow")
    ],
    task: Annotated[str, typer.Option("--task", "-t", help="task to train")],
    input_files: Annotated[list[str], typer.Argument(help="Input directory of files")],
    keep_log: Annotated[
        bool, typer.Option("--keep-log", help="Keep previous log files")
    ] = True,
):
    """Train a specific machine learning task in the Collectra workflow.

    Executes the training process for a specified task within the workflow,
    using the provided input files for training data. Generates timestamped
    log files during training and optionally cleans them up afterward.

    Args:
        workflow (Path): Path to the workflow configuration file containing the task.
        task (str): Name of the specific task to train within the workflow.
        input_files (list[str]): List of input file paths or directories containing
            training data for the task.
        keep_log (bool, optional): Whether to preserve training log files after
            completion. Defaults to False (logs are deleted).

    Raises:
        Exception: If the task cannot be trained due to invalid task name,
            missing input files, or training process failures.
    """
    try:
        log = Path.cwd() / f"{task}_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        config = {"input": input_files, "log_dir": log, "project": Path.cwd() / f"{workflow.name}-{task}"}
        pipeline = resolve_workflow_path(workflow)
        pipeline.train(task, **config)
        pipeline.save()
        if not keep_log:
            shutil.rmtree(log, ignore_errors=True)
            log_cache = Path(f"{log}.cache")
            if log_cache.exists():
                os.remove(log_cache)
    except Exception as e:
        traceback.print_exc()


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
@app.command()
def run(
    workflow: Annotated[
        Path, typer.Option("--workflow", "-w", help="path to workflow")
    ],    
    inputs: Annotated[list[Path], typer.Argument(help="Input files for the workflow")],
    task: Annotated[str, typer.Option("--task", "-t", help="task to run")] = "",
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="output directory")
    ] = None,
    single: Annotated[
        bool, typer.Option("--single", help="Runs only the designated task")
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enables verbose output during workflow execution")
    ] = False,
):
    """Execute a Collectra workflow or specific task within a workflow.

    Runs the specified workflow starting from either a specific task or from
    the root tasks if no task is specified. Additional command-line arguments
    are parsed and passed to the workflow execution as input parameters.

    Args:
        workflow (Path): Path to the workflow configuration file to execute.
        ctx (Context): Typer context containing additional command-line arguments.
        task (str, optional): Name of the specific task to run. If empty, runs
            from root tasks in the workflow. Defaults to empty string.
        output (Path, optional): Directory path where workflow results will be
            saved. Defaults to current working directory.

    Raises:
        Exception: If the workflow execution fails due to invalid workflow file,
            missing task, or runtime errors during execution.
    """
    try:
        data: dict[str, str | bool | list[str] | Path | list[Path]] = dict()
        data["single"] = single
        if output:
            data["output"] = str(output)
        data["files"] = list()
        pipeline = resolve_workflow_path(workflow)        
        for input_path in inputs:
            input_path = Path(input_path)            
            if input_path.is_file() or (input_path.is_dir() and input_path.suffix.lower() == f".{pipeline.ext}"):
                data["files"].append(input_path)
            elif input_path.is_dir():
                data["files"] += [file for file in input_path.rglob("*") if file.is_file() or (file.is_dir() and file.suffix.lower() == pipeline.ext)]               
        pipeline(task, **data, verbose=verbose)
    except Exception as e:
        traceback.print_exc()


@app.command()
def view(
    file: Path = typer.Option("--file", help="The collectra result file to be viewed")
):
    try:
        editor = Editor(file)
        editor.view()
    except Exception as e:
        traceback.print_exc()

# @app.command()
# def cluster_images(    
#     inputs: List[Path],
#     label: str = typer.Option("image_clustering", help="The label for the clustering task."),
#     format: str = typer.Option("grapto", help="The output format for the clustered data (e.g., 'parquet', 'csv')."),
#     output_df: Path = Path.cwd() / Path("clustered_images.parquet"),
#     output_html:Path = None,
#     model:str = typer.Option("vgg19", help="The name of the torchvision model to use (see https://pytorch.org/vision/stable/models.html#)."),
#     max_images:int = None,
#     algorithm:str = "SPECTRAL",
#     n_clusters:int = 20,
#     batch_size:int = 1,
#     thumbnail_width:int = 256,
#     thumbnail_height:int = 256,
#     force:bool = False,
#     force_features:bool = False,
#     force_pca:bool = False,
#     force_cluster:bool = False,
#     force_thumbnails:bool = False,
# ):
#     try:
#         input_images = []
#         for path in inputs:
#             path = Path(path)
#             # If it is a text file, then read each line as an image
#             if path.is_dir() and (path / "results.yaml").exists():
#                 input_images.append(path)
#             elif path.suffix.lower() == ".txt":
#                 with open(path) as f:
#                     paths_in_file = [Path(line.strip()) for line in f.readlines()]
#                     input_images += [x for x in paths_in_file if Path(x).is_dir() and (Path(x) / "results.yaml").exists()]
#             elif path.is_dir() and not (path / "results.yaml").exists():                
#                 for image_path in path.rglob(f"*.{format}"):
#                     if image_path.is_dir() and (image_path / "results.yaml").exists():
#                         input_images.append(image_path)

#         skipped_images = []

#         with tempfile.TemporaryDirectory() as temp_dir:
#             temp_path = Path(temp_dir)

#             for img_path in input_images:
#                 with change_dir(img_path):
#                     with open("results.yaml", "r") as f:
#                         results = yaml.safe_load(f)
#                     if label not in results:
#                         print(f"Label {label} not found in {img_path}/results.yaml. Skipping...")
#                         skipped_images.append(img_path)
#                         continue
#                     values = results[label] if isinstance(results[label], list) else [results[label]]                                        
#                     for item in values:
#                         if "path" not in item and "data" not in item:
#                             print(f"Invalid format in {img_path}/results.yaml for label {label}. No data or path found. Skipping...")
#                             skipped_images.append(img_path)
#                             continue
#                         folder_path = Path(temp_path / f"{img_path.stem}")
#                         folder_path.mkdir(parents=True, exist_ok=True)
#                         item["data"] = Path(item.pop("path")) if "path" in item else Path(item["data"])
#                         cls_ = load_class_from_string(item.pop("type"))
#                         item["name"] = label
#                         img = cls_(**item)                                                
#                         img.pil().save(folder_path / f"{img.id}.{img.ext.lower()}")

#             file_inputs = [file for file in temp_path.rglob("*") if file.is_file()]                      
        
#             images, features = clusterer(
#                 file_inputs,
#                 output_df,
#                 output_html,
#                 model,
#                 max_images,
#                 algorithm,
#                 n_clusters,
#                 batch_size,
#                 thumbnail_width,
#                 thumbnail_height,
#                 force,
#                 force_features,
#                 force_pca,
#                 force_cluster,
#                 force_thumbnails,
#             )            

#             model_name = model.replace("/", "_")

#             for index, feature in enumerate(features):
#                 feature_img_name = re.sub(r".*/", "", str(images[index].parent))
#                 input_image = next((img_path for img_path in input_images if img_path.stem == feature_img_name), None)                
#                 if input_image is None:
#                     print(f"Could not find original image for clustered image {images[index].stem}. Skipping metadata update.")
#                     continue
#                 with change_dir(input_image):
#                     with open("results.yaml", "r") as f:
#                         results = yaml.safe_load(f)
#                     if label not in results:
#                         print(f"Label {label} not found in {input_image}/results.yaml during metadata update. Skipping...")
#                         continue
#                     values = results[label] if isinstance(results[label], list) else [results[label]]                                        
#                     for item in values:
#                         if "path" not in item and "data" not in item:
#                             print(f"Invalid format in {input_image}/results.yaml for label {label} during metadata update. No data or path found. Skipping...")
#                             continue
#                         item_id = item.get("id", None)                        
#                         if item_id == images[index].stem or item_id is None:
#                             if item_id is None:                                
#                                 item["id"] = item_id
#                             # Save feature embedding as a .npy file
#                             feature_path = Path(f"{model_name}/{label}/{item_id}/")
#                             feature_path.mkdir(parents=True, exist_ok=True)
#                             with open(feature_path / f"{model_name}_{label}_{item['id']}.npy", "wb") as f:
#                                 np.save(f, feature)
#                             if "embedding" not in item:
#                                 item["embedding"] = []
#                             item["embedding"].append(str(feature_path))       

#                     if len(values) == 1:
#                         results[label] = values[0]               

#                     with open("results.yaml", "w") as f:
#                         for file_key, data in results.items():
#                             yaml.dump({file_key: data}, f, sort_keys=False, allow_unicode=True)
#                             f.write("\n")       


#         print(f"Clustering completed. Processed {len(images)} images.")
#         if skipped_images:
#             print(f"Skipped {len(skipped_images)} images due to missing label '{label}':")
#             for skipped in skipped_images:
#                 print(f" - {skipped}")
        
#     except Exception as e:
#         traceback.print_exc()
#         print(f"Error during clustering: {e}")

if __name__ == "__main__":
    app()
