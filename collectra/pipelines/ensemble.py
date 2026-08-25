"""Ensemble processing module for Collectra pipelines.

This module provides functionality for ensembling multiple collectra result files
into a single combined result using various techniques:
- Weighted Box Fusion (WBF) for ImageCrop ensembling
- Edit distance clustering for Text ensembling
- IoU-based matching for provenance tracking

Classes:
    EnsembleProcessor: Main class for processing ensemble operations

Functions:
    calculate_iou: Calculate Intersection over Union between two boxes
    find_centroid_text: Find the medoid text in a cluster
    cluster_texts: Cluster similar texts using single-linkage clustering
    find_contributing_boxes: Find source boxes that contributed to a fused box
"""

__all__ = [
    "EnsembleProcessor",
    "calculate_iou",
    "find_centroid_text",
    "cluster_texts",
    "find_contributing_boxes",
]

import datetime
import shutil
from pathlib import Path

import editdistance
import yaml
from ensemble_boxes import weighted_boxes_fusion
from rich.progress import track

from ..types.base import Artefact, DataNode
from ..types.images import Image, ImageCrop
from ..types.texts import Text

# =============================================================================
# Pure Utility Functions (no class dependencies)
# =============================================================================


def calculate_iou(box1: list[float], box2: list[float]) -> float:
    """Calculate IoU (Intersection over Union) between two boxes.

    Args:
        box1: Box 1 in format [x1, y1, x2, y2]
        box2: Box 2 in format [x1, y1, x2, y2]

    Returns:
        IoU value between 0 and 1
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


def find_centroid_text(texts: list[str]) -> str:
    """Find the text with minimum total edit distance to all others (medoid).

    Args:
        texts: List of text strings to find centroid from

    Returns:
        The text that is most representative of the cluster
    """
    if len(texts) == 1:
        return texts[0]

    min_total_distance = float("inf")
    centroid = texts[0]

    for candidate in texts:
        total_distance = sum(editdistance.eval(candidate, other) for other in texts)
        if total_distance < min_total_distance:
            min_total_distance = total_distance
            centroid = candidate

    return centroid


def cluster_texts(
    texts: list[str],
    provenance: list[str],
    num_clusters: int,
    similarity_threshold: float,
) -> list[dict]:
    """Cluster similar texts and find centroid for each cluster.

    Uses single-linkage clustering: a text joins a cluster if it's similar
    to ANY existing member, allowing transitive similarity chains.

    Args:
        texts: List of text strings to cluster
        provenance: List of source references for each text
        num_clusters: Maximum number of clusters to create
        similarity_threshold: Minimum similarity (0-1) to consider texts as same cluster

    Returns:
        List of cluster dicts with 'centroid' and 'contributors' keys
    """
    clusters = []
    used: set[int] = set()

    while len(used) < len(texts) and len(clusters) < num_clusters:
        for i, text in enumerate(texts):
            if i in used:
                continue
            # Start new cluster
            used.add(i)
            cluster = [text]
            prov = [provenance[i]]

            # Iteratively find all similar texts (single-linkage)
            # Keep expanding until no more texts can be added
            for j, other in enumerate(texts):
                if j in used:
                    continue
                max_len = max(len(text), len(other)) or 1
                similarity = 1 - (editdistance.eval(text, other) / max_len)
                if similarity >= similarity_threshold:
                    cluster.append(other)
                    prov.append(provenance[j])
                    used.add(j)

            centroid = find_centroid_text(cluster)
            clusters.append(
                {
                    "centroid": centroid,
                    "contributors": prov,
                }
            )
            break  # Restart fresh to find next cluster

    return clusters


def find_contributing_boxes(
    fused_box: list[float],
    source_boxes: list[list[float]],
    provenance: list[str],
    iou_threshold: float = 0.5,
) -> list[str]:
    """Find source boxes that contributed to a fused box via IoU.

    Args:
        fused_box: The fused box in format [x1, y1, x2, y2]
        source_boxes: List of source boxes
        provenance: List of source references for each box
        iou_threshold: Minimum IoU to consider as contributor

    Returns:
        List of source references that contributed to the fused box

    Raises:
        ValueError: If no contributing boxes are found
    """
    contributors = list()
    for box, prov in zip(source_boxes, provenance):
        if calculate_iou(fused_box, box) >= iou_threshold:
            contributors.append(prov)
    if len(contributors) == 0:
        raise ValueError("No contributing boxes found for fused box")
    return list(contributors)


# =============================================================================
# EnsembleProcessor Class
# =============================================================================


class EnsembleProcessor:
    """Processor for ensembling multiple collectra result files.

    This class handles the ensemble logic for combining results from multiple
    sources into a single merged result with provenance tracking.

    Attributes:
        node_manager: Node manager for resolving data nodes
        ext: File extension for collectra result folders
        name: Workflow name
        version: Workflow version
    """

    def __init__(self, node_manager, ext: str, name: str, version: str):
        """Initialize the EnsembleProcessor.

        Args:
            node_manager: Node manager for resolving data nodes
            ext: File extension for collectra result folders
            name: Workflow name
            version: Workflow version
        """
        self.node_manager = node_manager
        self.ext = ext
        self.name = name
        self.version = version

    def ensemble(self, folders: list[Path], output: Path) -> None:
        """Create ensembled files from source folders.

        Args:
            folders: List of source folders containing collectra results
            output: Output folder path for ensembled results
        """
        file_linkage = self.create_file_linkage(folders)
        layers: list[list[DataNode]] = self.generate_ensembling_layers()
        output_folder = Path(output)
        for filename, source_paths in track(
            file_linkage.items(), description="Ensembling files..."
        ):
            self._reset_nodes()
            self.ensemble_single_file(filename, source_paths, layers, output_folder)
        with open(output_folder / "linkage.yml", "w") as f:
            yaml.dump(file_linkage, f, sort_keys=False)

    def create_file_linkage(self, folders: list[Path]) -> dict[str, list[str]]:
        """Identify unique filenames across folders and create linkage.

        Args:
            folders: List of source folders to scan

        Returns:
            Dictionary mapping filename to list of source paths
            Example: {"file1": ["folder1/file1.collectra", "folder2/file1.collectra"]}
        """
        file_linkage: dict[str, list[str]] = {}

        # Scan all folders for files with the target extension
        for folder in folders:
            if not folder.exists():
                continue

            for item in folder.iterdir():
                if item.is_dir() and item.name.endswith(self.ext):
                    # Extract the filename without extension
                    filename = item.name
                    file_linkage.setdefault(filename, []).append(str(item))

        return file_linkage

    def generate_ensembling_layers(self) -> list[list[DataNode]]:
        """Generate layers of data nodes for ensemble processing.

        Returns:
            List of layers, where each layer contains DataNode objects
            that should be processed together
        """
        layers = []
        all_names = self.node_manager.get_node_names()
        first_node = self.node_manager.resolve_node(all_names[0])
        root_nodes = self.node_manager.get_parents(first_node)
        layers.append(root_nodes)
        for task_node in self.node_manager.get_task_nodes():
            output_nodes = self.node_manager.get_children_data(task_node)
            layers.append(output_nodes)
        return layers

    def ensemble_single_file(
        self,
        filename: str,
        source_paths: list[str],
        layers: list[list[DataNode]],
        output_folder: Path,
    ) -> None:
        """Ensemble a single file from source paths and save to output path.

        Args:
            filename: Name of the file being ensembled
            source_paths: List of paths to source collectra folders
            layers: List of data node layers to process
            output_folder: Output folder for the ensembled result
        """
        output_path = output_folder / filename
        output_path.mkdir(parents=True, exist_ok=True)
        ensembled_data = dict()
        for layer in layers:
            for data_node in layer:
                ensembled_data[data_node.name] = self.ensemble_at_layer(
                    data_node, source_paths, ensembled_data
                )
        full_data = dict()
        full_data["collectra_results_metadata"] = {
            "workflow": self.name,
            "version": self.version,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        full_data.update(ensembled_data)
        with open(output_path / "results.yaml", "w") as f:
            for key, value in ensembled_data.items():
                if value:
                    yaml.dump({key: value}, f, sort_keys=False)
                    f.write("\n")
        self.copy_artifact_files(Path(source_paths[0]), output_path)

    def ensemble_at_layer(
        self, data_node: DataNode, source_paths: list[str], ensembled_data: dict
    ) -> dict | list[dict]:
        """Ensemble data at a specific layer from source paths.

        Args:
            data_node: The data node to ensemble
            source_paths: List of source paths to read data from
            ensembled_data: Previously ensembled data for reference

        Returns:
            Ensembled data as dict or list of dicts
        """
        data_node.ensemble = True
        for source_path in source_paths:
            data_node.process(data_node.name, Path(source_path), skip_type_check=True)
        label_data = self.ensemble_data_node(data_node, ensembled_data)
        if label_data is None:
            return []
        if isinstance(label_data, list) and len(label_data) == 1:
            return label_data[0]
        else:
            return label_data

    def ensemble_data_node(
        self, data_node: DataNode, ensembled_data: dict
    ) -> list[dict] | dict | None:
        """Ensemble a data node based on its item types.

        Args:
            data_node: The data node containing ensemble items
            ensembled_data: Previously ensembled data for parent resolution

        Returns:
            Ensembled result or None if no items
        """
        items: dict[str, Artefact] = data_node.ensemble_items

        if not items:
            return None

        if all(type(item) == Image for item in items.values()):
            first_item: Image = list(items.values())[0]
            return {
                "type": "collectra.Image",
                "id": f"{data_node.name}_ensemble",
                "data": first_item.get_path().name,
                "ensemble": list(items.keys()),
            }

        elif all(type(item) == ImageCrop for item in items.values()):
            return self.ensemble_image_crops(data_node, ensembled_data)

        elif all(type(item) == Text for item in items.values()):
            return self.ensemble_text(data_node, ensembled_data)

        else:
            raise Warning(
                f"Ensembling not supported for data type in node {data_node.name}"
            )

    def ensemble_image_crops(
        self,
        data_node: DataNode,
        ensembled_data: dict,
    ) -> dict | list[dict]:
        """Ensemble ImageCrop items using Weighted Box Fusion.

        Args:
            data_node: The data node containing ImageCrop items
            ensembled_data: Previously ensembled data for parent resolution

        Returns:
            Ensembled ImageCrop result(s)
        """
        sources = list(
            set([key.split("::")[0] for key in data_node.ensemble_items.keys()])
        )

        source_crops = dict()

        for idx, item in data_node.ensemble_items.items():
            for source in sources:
                if not idx.startswith(source):
                    continue
                source_crops.setdefault(source, []).append(item)

        # Collect boxes for WBF
        boxes_per_source: list[list[list[float]]] = []
        scores_per_source: list[list[float]] = []
        labels_per_source: list[list[int]] = []
        prov_per_source: list[list[str]] = []
        data_source: str = ""

        for crops in source_crops.values():
            src_boxes, src_scores, src_labels, src_prov = [], [], [], []
            for crop in crops:
                # Convert center-format to corner-format (normalized 0-1)
                x1 = max(0, crop.x_center - crop.width_relative / 2)
                y1 = max(0, crop.y_center - crop.height_relative / 2)
                x2 = min(1, crop.x_center + crop.width_relative / 2)
                y2 = min(1, crop.y_center + crop.height_relative / 2)

                src_boxes.append([x1, y1, x2, y2])
                src_scores.append(1.0)
                src_labels.append(0)
                src_prov.append(crop.id)
                if data_source == "":
                    data_source = crop.get_path().name
                else:
                    if data_source == crop.get_path().name:
                        continue
                    raise ValueError("Mismatched data sources in crops for ensembling")

            boxes_per_source.append(src_boxes)
            scores_per_source.append(src_scores)
            labels_per_source.append(src_labels)
            prov_per_source.append(src_prov)

        # WBF call
        weights = [1] * len(boxes_per_source)
        fused_boxes, _, _ = weighted_boxes_fusion(
            boxes_per_source,
            scores_per_source,
            labels_per_source,
            weights=weights,
            iou_thr=0.5,
            skip_box_thr=0.0,
        )

        # Build result
        result = []
        flat_prov = [p for sp in prov_per_source for p in sp]
        flat_boxes = [b for sb in boxes_per_source for b in sb]

        for i, fused_box in enumerate(fused_boxes):
            # Convert back to center format
            x1, y1, x2, y2 = fused_box
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            width_rel = x2 - x1
            height_rel = y2 - y1

            # Find contributing boxes via IoU
            ensemble_refs = find_contributing_boxes(
                list(fused_box), flat_boxes, flat_prov, iou_threshold=0.5
            )

            ensemble_ref_nodes = [
                data_node.ensemble_items[key]
                for key in data_node.ensemble_items.keys()
                if key in ensemble_refs
            ]

            # Resolve parents from contributing crops
            ensembled_parents = self.resolve_ensemble_parents(
                ensemble_ref_nodes, ensembled_data
            )

            result.append(
                {
                    "type": "collectra.ImageCrop",
                    "id": f"{data_node.name}_ensembled_{i+1}",
                    "data": data_source,
                    "ensemble": ensemble_refs,
                    "parents": ensembled_parents,
                    "x_center": float(x_center),
                    "y_center": float(y_center),
                    "width_relative": float(width_rel),
                    "height_relative": float(height_rel),
                }
            )

        return result[0] if len(result) == 1 else result

    def ensemble_text(
        self, data_node: DataNode, ensembled_data: dict
    ) -> dict | list[dict]:
        """Ensemble Text items using edit distance clustering.

        Args:
            data_node: The data node containing Text items
            ensembled_data: Previously ensembled data for parent resolution

        Returns:
            Ensembled Text result(s) with provenance
        """
        sources = list(
            set([key.split("::")[0] for key in data_node.ensemble_items.keys()])
        )
        source_texts: dict[str, list] = {}

        for idx, item in data_node.ensemble_items.items():
            for source in sources:
                if idx.startswith(source):
                    source_texts.setdefault(source, []).append(item)

        # Flatten texts with provenance
        flat_texts = []
        flat_prov = []
        src_for_clustering = dict()
        for key, texts in source_texts.items():
            for text in texts:
                src_for_clustering.setdefault(key, []).append(text)
                flat_texts.append(text.data)
                flat_prov.append(text.id)

        num_clusters = 0
        for items in src_for_clustering.values():
            if len(items) > num_clusters:
                num_clusters = len(items)

        # Cluster similar texts and find centroids
        clusters = cluster_texts(
            flat_texts, flat_prov, num_clusters, similarity_threshold=0.7
        )

        # Build results
        result = []
        for i, cluster in enumerate(clusters):
            ensemble_refs = cluster["contributors"]

            ensemble_ref_nodes = [
                data_node.ensemble_items[key]
                for key in data_node.ensemble_items.keys()
                if key in ensemble_refs
            ]

            ensembled_parents = self.resolve_ensemble_parents(
                ensemble_ref_nodes, ensembled_data
            )

            result.append(
                {
                    "type": "collectra.Text",
                    "id": f"{data_node.name}_ensembled_{i+1}",
                    "ensemble": ensemble_refs,
                    "parents": ensembled_parents,
                    "data": cluster["centroid"],
                }
            )

        return result[0] if len(result) == 1 else result

    def resolve_ensemble_parents(
        self,
        ensemble_ref_nodes: list[Artefact],
        ensemble_data: dict,
    ) -> list[str]:
        """Resolve parent references from ensemble data.

        Args:
            ensemble_ref_nodes: List of data objects with parent references
            ensemble_data: Previously ensembled data to search for parents

        Returns:
            List of parent IDs or single parent ID if only one
        """
        ensembled_parents = set()
        for ensemble in ensemble_ref_nodes:
            parents = ensemble.parents
            for parent in parents:
                for key in ensemble_data:
                    if isinstance(ensemble_data[key], list):
                        for item in ensemble_data[key]:
                            if "ensemble" in item and parent in item["ensemble"]:
                                ensembled_parents.add(item["id"])
                    else:
                        if (
                            "ensemble" in ensemble_data[key]
                            and parent in ensemble_data[key]["ensemble"]
                        ):
                            ensembled_parents.add(ensemble_data[key]["id"])
        ensembled_parents = list(ensembled_parents)
        ensembled_parents = (
            ensembled_parents[0] if len(ensembled_parents) == 1 else ensembled_parents
        )
        return ensembled_parents

    def copy_artifact_files(self, source_path: Path, dest_path: Path) -> None:
        """Copy artifact files from first source to ensemble folder.

        Copies all files except results.yaml from the source collectra folder
        to the destination folder.

        Args:
            source_path: Path to first source collectra folder
            dest_path: Path to ensemble collectra folder
        """
        if not source_path.exists():
            return
        for item in source_path.iterdir():
            # Skip results.yaml
            if item.name == "results.yaml":
                continue
            # Copy artifact files
            if item.is_file():
                dest_file = dest_path / item.name
                shutil.copy2(item, dest_file)

    def _reset_nodes(self) -> None:
        """Reset all data nodes for fresh processing."""
        self.node_manager.reset_all_nodes()
