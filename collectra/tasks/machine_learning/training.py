"""Opt-in utilities for training task implementations."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Callable

from collectra.types.base import ArtefactNode
from collectra.types.images import Image, ImageCrop
from collectra.utils import change_dir

from ...logger import get_logger

logger = get_logger(__name__)


def prepare_object_detection_inputs(
    processed_inputs: list,
    processed_parents: list,
    input_maps: dict[str, list],
    parent_input_maps: dict[str, list],
) -> list:
    """Attach source images to detection crops and return the task inputs."""
    for item in processed_inputs:
        if type(item) is Image:
            continue
        if type(item) is not ImageCrop:
            raise TypeError(f"Input {item.id} is not an ImageCrop")
        if isinstance(item.parents, str):
            parent_id = item.parents
        elif isinstance(item.parents, list) and len(item.parents) == 1:
            parent_id = item.parents[0]
        else:
            continue
        parent = next(
            (
                candidate
                for candidate in processed_parents
                if candidate.id == parent_id and item.data == candidate.data
            ),
            None,
        )
        if not isinstance(parent, Image):
            raise ValueError(
                f"Image parent {parent_id!r} for input {item.id!r} was not found"
            )
        item.add_source_parent(parent)
    return processed_inputs


def _training_files(task, inputs: list[str]) -> list[Path]:
    pipeline = _pipeline_for(task)
    files = []
    for value in inputs:
        path = Path(value)
        if path.is_dir():
            files.extend(path.glob(f"*.{pipeline.ext}"))
        elif path.suffix == f".{pipeline.ext}":
            files.append(path)
    return files


def _prepare_data(task, inputs: list[str], nodes: list[ArtefactNode]):
    processed = []
    input_maps = {}
    for item_file in _training_files(task, inputs):
        input_maps[item_file.name] = ArtefactNode.batch_process(item_file, nodes)
        processed.extend(input_maps[item_file.name])
    return processed, input_maps


def _pipeline_for(task):
    if task.pipeline is None:
        raise RuntimeError(f"Task {task.name!r} is not attached to a pipeline")
    return task.pipeline


def save_training_result(task, results, *, log: str, **kwargs) -> None:
    """Copy the best model and optional class metadata into the pipeline."""
    pipeline = _pipeline_for(task)
    best_model_path = results.save_dir / "weights" / "best.pt"
    if not best_model_path.exists():
        raise FileNotFoundError(f"Best model file not found at {best_model_path}")
    logger.info("Training metrics: %s", results.results_dict)
    current_model = pipeline.data.get(task.name, {}).get("model", "")
    if best_model_path.name == current_model:
        return
    new_model_path = f"{task.name}-{Path(log).name}.pt"
    shutil.copy(best_model_path, new_model_path)
    pipeline.data[task.name]["model"] = new_model_path
    classes_src = best_model_path.parent / "classes.json"
    if classes_src.exists():
        shutil.copy(classes_src, f"{Path(new_model_path).stem}.classes.json")


def train_from_files(
    task,
    inputs: list[str],
    trainer: Callable,
    *,
    prepare_inputs: Callable | None = None,
    **kwargs,
):
    """Resolve pipeline artefacts and invoke a concrete backend trainer."""
    pipeline = _pipeline_for(task)
    task_node = pipeline.node_manager.resolve_node(task.name)
    children = pipeline.node_manager.get_children_artefact(task_node)
    parents = pipeline.node_manager.get_parents_artefact(task_node)
    kwargs["classes"] = kwargs.get("classes", [child.name for child in children])
    kwargs = pipeline._merge_task_params(task.name, kwargs)
    processed_inputs, input_maps = _prepare_data(task, inputs, children)
    processed_parents, parent_input_maps = _prepare_data(task, inputs, parents)
    if prepare_inputs is not None:
        processed_inputs = prepare_inputs(
            processed_inputs,
            processed_parents,
            input_maps,
            parent_input_maps,
        )
    with change_dir(pipeline.path):
        results = trainer(*processed_inputs, **kwargs)
        save_training_result(task, results, **kwargs)
    return results


def run_training_command(
    task,
    inputs: list[str],
    trainer: Callable,
    *,
    keep_log: bool,
    validation: str,
    exclude: str,
    prepare_inputs: Callable | None = None,
    **kwargs,
):
    """Run the common filesystem lifecycle requested by a concrete command."""
    pipeline = _pipeline_for(task)
    log = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = train_from_files(
        task,
        inputs,
        trainer,
        prepare_inputs=prepare_inputs,
        project=f"{pipeline.path.name}-{task.name}",
        log=log,
        base_folder=Path.cwd(),
        validation=validation,
        exclude=exclude,
        **kwargs,
    )
    pipeline.save(task.name)
    if not keep_log:
        shutil.rmtree(log, ignore_errors=True)
    return results
