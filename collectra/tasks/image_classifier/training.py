"""Shared preparation of labelled classifier inputs."""

from collectra.types.images import Image
from collectra.types.links import Link


def prepare_classification_inputs(
    processed_inputs: list,
    processed_parents: list,
    input_maps: dict[str, list],
    parent_input_maps: dict[str, list],
) -> list:
    """Use parent images labelled by their child Link node names."""
    labeled_parents = []
    for file_name, children in input_maps.items():
        parents = parent_input_maps.get(file_name, [])
        artefacts = {item.id: item for item in [*parents, *children]}
        for item in artefacts.values():
            if type(item) is Link:
                item.bind(artefacts)
        for child in children:
            if type(child) is not Link:
                raise TypeError(
                    f"Classifier label {child.id!r} in {file_name!r} must be "
                    f"a Link, got {type(child).__name__}"
                )
            if not child.parents:
                raise ValueError(
                    f"Classifier Link {child.id!r} in {file_name!r} has no parent"
                )
            parent_id = (
                child.parents if isinstance(child.parents, str) else child.parents[0]
            )
            parent = artefacts.get(parent_id)
            if parent is None:
                raise ValueError(
                    f"Classifier Link {child.id!r} in {file_name!r} points to "
                    f"missing parent {parent_id!r}; available parent IDs: "
                    f"{[item.id for item in parents]}"
                )
            child.target = parent
            try:
                target = child.resolve()
            except RuntimeError as error:
                raise ValueError(
                    f"Classifier Link {child.id!r} in {file_name!r} cannot "
                    f"resolve parent chain: {error}"
                ) from error
            if not isinstance(target, Image):
                raise TypeError(
                    f"Classifier Link {child.id!r} resolves to "
                    f"{type(target).__name__}, expected Image or ImageCrop"
                )
            child.partition = target.partition
            labeled_parents.append(child)
    return labeled_parents
