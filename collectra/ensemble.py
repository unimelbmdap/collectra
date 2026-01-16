__all__ = ["Ensembler"]

import logging
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import editdistance
import yaml
from ensemble_boxes import weighted_boxes_fusion
from rich import print
from rich.progress import track

from .types.images import Image, ImageCrop
from .types.texts import Text
from .utils import load_class_from_string

logger = logging.getLogger(__name__)


@dataclass
class Ensembler:
    folders: list[Path]
    output: Path
    extension: str
    tmp_ensembled: dict = field(default_factory=dict)

    def ensemble(self) -> int:
        """
        Main ensemble workflow orchestration.

        Steps:
        1. Create file linkage across folders
        2. Identify unique labels
        3. Create ensemble output folder structure
        4. Process each file and ensemble its labels
        5. Write results.yaml with metadata
        """
        # Create file linkage, identify labels, create folder structure
        file_linkage = self._create_file_linkage()

        if not file_linkage:
            logger.warning("No files found for ensembling.")
            return 0

        # Identify unique labels
        unique_labels = self._identify_unique_labels(file_linkage)

        # Create ensemble folder structure
        self._create_ensemble_folder_structure(file_linkage)

        # Save linkage to link.yaml
        self._save_link_yaml(file_linkage, unique_labels)

        # Process labels for each file
        for filename, source_paths in track(
            file_linkage.items(), description="Ensembling files..."
        ):
            self.tmp_ensembled = self._retrieve_label_data_for_file(
                filename, source_paths, unique_labels
            )
            workflow_name = "Ensemble"
            # Process each data type
            ensembled_labels = dict()
            crop_id_mapping = dict()
            for data_type in [Image, ImageCrop, Text]:
                # Process each label in file
                for label_name, label_items in self.tmp_ensembled.items():
                    self._loop_label_processing(
                        ensembled_labels,
                        label_name,
                        label_items,
                        data_type,
                        crop_id_mapping,
                    )
            self._write_results_yaml(
                self.output / filename,
                ensembled_labels,
                self._generate_metadata(
                    source_paths,
                    original_workflow=workflow_name,
                ),
            )
            self._copy_artifact_files(Path(source_paths[0]), self.output / filename)

        return len(file_linkage)

    def _loop_label_processing(
        self,
        ensembled_labels: dict,
        label_name: str,
        label_items: dict,
        check_type: type,
        crop_id_mapping: dict[str, list[str]] | None = None,
    ) -> None:
        label_type: str = label_items.get("type", "")
        if load_class_from_string(label_type) is not check_type:
            return None
        ensembled_label = self._process_file_label(
            label_name,
            label_items,
            label_type,
            crop_id_mapping=crop_id_mapping,
        )
        if crop_id_mapping is not None and ensembled_label is not None:
            self._update_crop_id_mapping(
                ensembled_label,
                crop_id_mapping,
            )
        if ensembled_label:
            ensembled_labels[label_name] = ensembled_label

    def _update_crop_id_mapping(
        self,
        ensembled_label: dict | list[dict],
        crop_id_mapping: dict[str, list[str]],
    ) -> None:
        if isinstance(ensembled_label, list):
            for item in ensembled_label:
                ensemble_ref = item["id"]
                for child in item.get("ensemble", []):
                    crop_id_mapping.setdefault(child, []).append(ensemble_ref)
        elif isinstance(ensembled_label, dict):
            ensemble_ref = ensembled_label["id"]
            for child in ensembled_label.get("ensemble", []):
                crop_id_mapping.setdefault(child, []).append(ensemble_ref)

    def _create_file_linkage(self) -> dict[str, list[str]]:
        """
        Identify unique filenames across folders and create linkage.

        Returns:
            Dictionary mapping filename to list of source paths
            Example: {"file1": ["folder1/file1.collectra", "folder2/file1.collectra"]}
        """
        file_linkage: Dict[str, List[str]] = {}

        # Scan all folders for files with the target extension
        for folder in self.folders:
            if not folder.exists():
                continue

            for item in folder.iterdir():
                if item.is_dir() and item.name.endswith(self.extension):
                    # Extract the filename without extension
                    filename = item.name

                    if filename not in file_linkage:
                        file_linkage[filename] = []

                    file_linkage[filename].append(str(item))

        return file_linkage

    def _identify_unique_labels(self, file_linkage: Dict[str, List[str]]) -> List[str]:
        """
        Identify all unique labels across all source files.

        TODO: Hard code for now, implement proper logic later.

        Args:
            file_linkage: Dictionary of file linkage

        Returns:
            List of all unique label names
        """
        return [
            "specimen_sheet",
            "primary_label_unoriented",
            "primary_label",
            "registration_number_image",
            "genus_image",
            "specific_epithet_image",
            "formation_image",
            "age_image",
            "locality_image",
            "collection_origin_image",
            "collector_image",
            "previous_number_image",
            "registration_number_draft",
            "genus_draft",
            "specific_epithet_draft",
            "formation_draft",
            "age_draft",
            "locality_draft",
            "collection_origin_draft",
            "collector_draft",
            "previous_number_draft",
            "registration_number",
            "genus",
            "specific_epithet",
            "formation",
            "age",
            "locality",
            "collection_origin",
            "collector",
            "previous_number",
        ]

    def _load_source_file_content(self, file_path: Path) -> Dict[str, Any]:
        """
        Load results.yaml from a source collectra file.

        Args:
            file_path: Path to source collectra folder

        Returns:
            Dictionary of results.yaml content
        """
        results_yaml = file_path / "results.yaml"

        if not results_yaml.exists():
            return {}

        with open(results_yaml, "r") as f:
            content = yaml.safe_load(f)
            return content if content else {}

    def _retrieve_label_data_for_file(
        self, filename: str, source_paths: List[str], unique_labels: List[str]
    ) -> Dict[str, Any]:
        """
        Retrieve label data from all sources for a specific file (Step 8, substeps 1-2).

        Substep 1: Create a file-level storage dictionary
        Substep 2: Iterate through each label and retrieve data from all sources

        Args:
            filename: Name of the ensemble file
            source_paths: List of source file paths for this filename
            unique_labels: List of all unique labels across all sources

        Returns:
            Dictionary with format:
            {
                "label_name": {
                    "type": "collectra.Image",
                    "items": {
                        "folder/file": [item1, item2, ...],
                        ...
                    }
                },
                ...
            }
        """
        # Substep 1: Create file-level storage dictionary
        storage_dict: Dict[str, Any] = {}

        # Substep 2: Iterate through each label
        for label_name in unique_labels:
            label_storage = {"items": {}}
            label_type = None

            # Retrieve label data from all source files
            for source_path in source_paths:
                source_content = self._load_source_file_content(Path(source_path))

                # Check if label exists in this source
                if label_name in source_content:
                    label_data = source_content[label_name]

                    # Normalize to list format (single item -> list)
                    if isinstance(label_data, list):
                        items = label_data
                    else:
                        items = [label_data]

                    # Store items for this source
                    label_storage["items"][source_path] = items

                    # Extract type from first item if not already set
                    if label_type is None and items and isinstance(items[0], dict):
                        label_type = items[0].get("type")

            # Only add to storage_dict if label has items in at least one source
            if label_storage["items"]:
                if label_type:
                    label_storage["type"] = label_type
                storage_dict[label_name] = label_storage

        return storage_dict

    def _validate_image_labels(self, label_data: Dict[str, Any]) -> str:
        """
        Validate that Image type labels have consistent data sources.

        Args:
            label_data: Dictionary with type and items from all sources

        Returns:
            The validated data source string

        Raises:
            ValueError: If Images reference different source files
        """
        data_sources: set = set()

        for source_path, items in label_data.get("items", {}).items():
            for item in items:
                if "data" in item:
                    data_sources.add(item["data"])

        if len(data_sources) > 1:
            raise ValueError(
                f"Image labels reference different sources: {data_sources}"
            )

        return data_sources.pop() if data_sources else ""

    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """
        Calculate IoU between two boxes [x1, y1, x2, y2].

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

    def _find_contributing_boxes(
        self,
        fused_box: List[float],
        source_boxes: List[List[float]],
        provenance: List[str],
        iou_threshold: float = 0.5,
    ) -> List[str]:
        """
        Find source boxes that contributed to a fused box via IoU.

        Args:
            fused_box: The fused box in format [x1, y1, x2, y2]
            source_boxes: List of source boxes
            provenance: List of source references for each box
            iou_threshold: Minimum IoU to consider as contributor

        Returns:
            List of source references that contributed to the fused box
        """
        contributors = []
        for box, prov in zip(source_boxes, provenance):
            if self._calculate_iou(fused_box, box) >= iou_threshold:
                contributors.append(prov)
        return contributors if contributors else ([provenance[0]] if provenance else [])

    def _ensemble_image_crops(
        self,
        label_data: Dict[str, dict],
        label_name: str,
        crop_id_mapping: Dict[str, list[str]] | None = None,
    ) -> dict | list[dict]:
        """
        Ensemble ImageCrop labels using Weighted Box Fusion.

        Uses WBF parameters:
        - iou_threshold: 0.5
        - skip_box_threshold: 0.0
        - weights: Equal weights across sources

        Args:
            label_data: Dictionary of ImageCrop items from each source
            label_name: Name of the label being ensembled
            crop_id_mapping: Mapping of source crop refs to ensembled crop IDs

        Returns:
            Ensembled ImageCrop(s) with format including:
            - type: "collectra.ImageCrop"
            - ensemble: List of source references (folder::id)
            - parents: List of resolved parent IDs from source crops
            - Fused coordinates (x_center, y_center, width_relative, height_relative)
        """

        items = label_data.get("items", {})

        if not items:
            return {}

        crop_id_mapping = crop_id_mapping or {}

        # Build lookup table: ensemble_ref -> (source_path, crop_object)
        ensemble_ref_lookup: Dict[str, tuple[str, dict]] = {}
        for source_path, crops in items.items():
            for crop in crops:
                ensemble_ref = f"{Path(source_path).parent}::{crop.get('id', '')}"
                ensemble_ref_lookup[ensemble_ref] = (source_path, crop)

        # Collect boxes per source for WBF
        boxes_per_source: List[List[List[float]]] = []
        scores_per_source: List[List[float]] = []
        labels_per_source: List[List[int]] = []
        prov_per_source: List[List[str]] = []
        data_source: str = ""

        for source_path, crops in items.items():
            folder_name = Path(source_path).parent
            src_boxes, src_scores, src_labels, src_prov = [], [], [], []

            for crop in crops:
                # Convert center-format to corner-format (normalized 0-1)
                x_c = crop["x_center"]
                y_c = crop["y_center"]
                w = crop["width_relative"]
                h = crop["height_relative"]

                x1 = max(0, x_c - w / 2)
                y1 = max(0, y_c - h / 2)
                x2 = min(1, x_c + w / 2)
                y2 = min(1, y_c + h / 2)

                src_boxes.append([x1, y1, x2, y2])
                src_scores.append(1.0)
                src_labels.append(0)
                src_prov.append(f"{folder_name}::{crop.get('id', '')}")
                if not data_source:
                    data_source = crop.get("data", "")

            boxes_per_source.append(src_boxes)
            scores_per_source.append(src_scores)
            labels_per_source.append(src_labels)
            prov_per_source.append(src_prov)

        # WBF call
        weights = [1] * len(boxes_per_source)
        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
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
            ensemble_refs = self._find_contributing_boxes(
                fused_box, flat_boxes, flat_prov, iou_threshold=0.5
            )

            # Resolve parents from contributing crops
            resolved_parents = self._resolve_ensemble_parents(
                ensemble_refs, ensemble_ref_lookup, crop_id_mapping
            )

            result.append(
                {
                    "type": "collectra.ImageCrop",
                    "id": f"{label_name}_ensembled_{i+1}",
                    "data": data_source,
                    "ensemble": ensemble_refs,
                    "parents": resolved_parents,
                    "x_center": float(x_center),
                    "y_center": float(y_center),
                    "width_relative": float(width_rel),
                    "height_relative": float(height_rel),
                }
            )

        return result[0] if len(result) == 1 else result

    def _resolve_ensemble_parents(
        self,
        ensemble_refs: List[str],
        ensemble_ref_lookup: Dict[str, tuple[str, dict]],
        crop_id_mapping: Dict[str, list[str]],
    ) -> List[str]:
        """
        Resolve parent IDs from contributing source crops to ensembled parent IDs.

        For each contributing crop, extracts its parent(s) and resolves them to
        ensembled IDs using crop_id_mapping. Returns deduplicated list of parent IDs.

        Args:
            ensemble_refs: List of source crop references (folder::id) that contributed
            ensemble_ref_lookup: Lookup table mapping ensemble_ref to (source_path, crop_object)
            crop_id_mapping: Mapping of source crop refs to ensembled crop IDs

        Returns:
            List of unique resolved parent IDs
        """
        parents: List[str] = []

        for ensemble_ref in ensemble_refs:
            # Skip if crop not found in lookup
            if ensemble_ref not in ensemble_ref_lookup:
                continue

            source_path, source_crop = ensemble_ref_lookup[ensemble_ref]

            # Get parents from source crop
            crop_parents = source_crop.get("parents", [])
            if not isinstance(crop_parents, list):
                crop_parents = [crop_parents] if crop_parents else []

            # Resolve each parent to ensembled ID
            for parent_id in crop_parents:
                # Build compound ID for lookup
                compound_id = f"{Path(source_path).parent}::{parent_id}"
                ensembled_ids = crop_id_mapping.get(compound_id, [])

                # Get ensembled ID or fallback to original (e.g., for Images)
                if isinstance(ensembled_ids, list) and ensembled_ids:
                    ensembled_id = ensembled_ids[-1]
                else:
                    ensembled_id = parent_id

                # Add to list if not already present (deduplicate)
                if ensembled_id not in parents:
                    parents.append(ensembled_id)

        return parents

    def _find_consensus_text(self, texts: List[str]) -> str:
        """
        Find text that minimizes total edit distance. Tiebreaker: shortest.

        Args:
            texts: List of text strings to find consensus from

        Returns:
            The consensus text string
        """
        if not texts:
            return ""
        if len(texts) == 1:
            return texts[0]

        # Compute total edit distance for each candidate
        scores = []
        for candidate in texts:
            total_dist = sum(editdistance.eval(candidate, other) for other in texts)
            scores.append((total_dist, len(candidate), candidate))

        # Sort by (total_distance, length) - min wins
        scores.sort(key=lambda x: (x[0], x[1]))
        return scores[0][2]

    def _ensemble_text(
        self,
        label_data: Dict[str, list],
        label_name: str,
        crop_id_mapping: Dict[str, list[str]] | None = None,
    ) -> Any:
        """
        Ensemble Text labels using Levenshtein distance.

        Algorithm:
        - Group text items by their ImageCrop parent(s)
        - For each group, select consensus text using edit distance
        - Consensus: minimizes total edit distance to all others
        - Tiebreaker: select shortest string

        Args:
            label_data: Dictionary of Text items from each source
            label_name: Name of the label being ensembled
            crop_id_mapping: Mapping of source crop refs to ensembled crop IDs

        Returns:
            Ensembled Text item(s) with format including:
            - type: "collectra.Text"
            - ensemble: List of source references (folder::id)
            - data: Consensus text string
            - parents: References to ensembled ImageCrop IDs
        """
        items = label_data.get("items", {})
        if not items:
            return []

        crop_id_mapping = crop_id_mapping or {}

        groups: dict[str, list[tuple[dict, list[str], list[str]]]] = dict()

        for source_path, texts in items.items():
            for text in texts:
                # Get the parent folder of this source file
                folder_path = Path(source_path).parent
                # Get the parents of this text item
                text_parents = text.get("parents", [])
                provenances = []
                parents = []
                for parent in text_parents:
                    # Build the provenance ID
                    compound_id = f"{folder_path}::{parent}"
                    provenances.append(compound_id)
                    # Given a provenance ID, find the ensembled ids and add to parents
                    grand_parents = crop_id_mapping.get(compound_id, [])
                    for grand_parent in grand_parents:
                        if grand_parent not in parents:
                            # Avoid duplicates
                            parents.append(grand_parent)
                crop_parent = self._find_nearest_crop_parent(text, source_path)
                ensemble_id = crop_id_mapping.get(f"{folder_path}::{crop_parent}", [])
                ensemble_id = (
                    ensemble_id[-1]
                    if isinstance(ensemble_id, list) and len(ensemble_id) > 0
                    else ensemble_id if ensemble_id else ""
                )
                if isinstance(ensemble_id, list) or not ensemble_id:
                    raise ValueError(
                        f"Could not find unique ensemble ID for text parent {crop_parent}"
                    )
                parents.append(ensemble_id)
                groups.setdefault(ensemble_id, []).append((text, provenances, parents))

        # Process each group
        result = []
        idx = 1

        for text_tuples in groups.values():
            texts, provenances, parents = list(), list(), list()
            for t in text_tuples:
                texts.append(t[0].get("data"))
                for provenance in t[1]:
                    if provenance not in provenances:
                        provenances.append(provenance)
                for parent in t[2]:
                    if parent not in parents:
                        parents.append(parent)

            # Find consensus text
            consensus = self._find_consensus_text(texts)

            result.append(
                {
                    "type": "collectra.Text",
                    "id": f"{label_name}_ensembled_{idx}",
                    "data": consensus,
                    "ensemble": provenances,
                    "parents": parents,
                }
            )
            idx += 1

        return result[0] if len(result) == 1 else result

    def _find_nearest_crop_parent(self, text_item: dict, source_path: str) -> str:
        parent_id = text_item.get("parents", "")
        nearest_parent_id = (
            parent_id[-1]
            if isinstance(parent_id, list) and len(parent_id) > 0
            else parent_id
        )
        found_parent = None
        while not found_parent:
            label = nearest_parent_id.split("-")[0]
            parents = (
                self.tmp_ensembled.get(label, {}).get("items", {}).get(source_path, [])
            )
            if not parents:
                # No more parents to check and didn't find one
                return ""
            next_parent = False
            for parent in parents:
                if not parent.get("id", "") == nearest_parent_id:
                    continue
                if parent.get("type", "") == "collectra.ImageCrop":
                    found_parent = parent
                    break
                next_parent = True
                next_parents = parent.get("parents", [])
                nearest_parent_id = (
                    next_parents[-1]
                    if isinstance(next_parents, list) and len(next_parents) > 0
                    else next_parents
                )
            if not next_parent:
                break
        return found_parent["id"] if found_parent else ""

    def _process_file_label(
        self,
        label_name: str,
        label_items: Dict[str, dict],
        label_type: str,
        crop_id_mapping: dict[str, list[str]] | None = None,
    ) -> list[dict] | dict | None:
        """
        Process a single label for ensembling based on its type.

        Handles:
        - Image: Pass-through with validation
        - ImageCrop: Weighted Box Fusion ensembling
        - Text: Edit distance consensus ensembling
        - Other types: Pass-through behavior

        Args:
            label_name: Name of the label
            label_items: Items from all sources for this label
            label_type: Type of label (collectra.Image, collectra.ImageCrop, collectra.Text)
            crop_id_mapping: Mapping of source crop refs to ensembled crop IDs for text

        Returns:
            Ensembled label data or pass-through item
        """

        items = label_items.get("items", {})
        if not items:
            return None

        if load_class_from_string(label_type) == Image:
            # Validate and pass-through
            data_source = self._validate_image_labels(label_items)
            # Return first item with validated data
            first_item = next(iter(items.values()))[0]
            return {
                "type": "collectra.Image",
                "id": first_item.get("id", label_name),
                "data": data_source,
                "ensemble": [
                    f"{Path(source_path).parent}::{first_item.get('id', '')}"
                    for source_path in items.keys()
                ],
            }

        elif load_class_from_string(label_type) == ImageCrop:
            return self._ensemble_image_crops(label_items, label_name, crop_id_mapping)

        elif load_class_from_string(label_type) == Text:
            return self._ensemble_text(label_items, label_name, crop_id_mapping)

        else:
            # Unknown type: pass-through first item
            first_source = next(iter(items.values()))
            return first_source[0] if first_source else None

    def _create_ensemble_folder_structure(
        self, file_linkage: Dict[str, List[str]]
    ) -> None:
        """
        Create output folder structure for ensemble results.

        Creates:
        - output/ (main ensemble folder)
        - output/link.yaml (file and label linkage)
        - output/[file1].collectra/
        - output/[file2].collectra/
        - etc.
        """
        # Create main ensemble output folder
        self.output.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for each ensemble file
        for filename in file_linkage.keys():
            ensemble_file_path = self.output / filename
            ensemble_file_path.mkdir(parents=True, exist_ok=True)

    def _generate_metadata(
        self, source_files: List[str], original_workflow: str
    ) -> Dict[str, Any]:
        """
        Generate metadata for results.yaml.

        Args:
            source_files: List of source file paths
            original_workflow: Original workflow name from source metadata

        Returns:
            Metadata dictionary with:
            - workflow: Original workflow name
            - version: 0.1.0
            - timestamp: Current ISO timestamp
        """
        return {
            "collectra_results_metadata": {
                "workflow": original_workflow,
                "version": "0.1.0",
                "timestamp": datetime.now().isoformat(),
                "ensemble_sources": source_files,
            }
        }

    def _write_results_yaml(
        self, file_path: Path, content: Dict[str, Any], metadata: Dict[str, Any]
    ) -> None:
        """
        Write results.yaml for an ensembled collectra file.

        Args:
            file_path: Path to output collectra folder
            content: Ensembled label content
            metadata: results.yaml metadata
        """
        # Combine metadata and content
        full_content = {
            "collectra_results_metadata": metadata["collectra_results_metadata"],
            **content,
        }

        results_yaml_path = file_path / "results.yaml"

        with open(results_yaml_path, "w") as f:
            yaml.dump(full_content, f, default_flow_style=False, sort_keys=False)

    def _copy_artifact_files(self, source_path: Path, dest_path: Path) -> None:
        """
        Copy artifact files (.jpg, .txt, etc.) from first source to ensemble folder.

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

    def _save_link_yaml(
        self, file_linkage: Dict[str, List[str]], labels: List[str]
    ) -> None:
        """
        Save file and label linkage to link.yaml.

        Format:
        ```yaml
        file_link:
          file1:
            - folder1/file1
            - folder2/file1
        labels:
          - label1
          - label2
        ```

        Args:
            file_linkage: Dictionary of file linkage
            labels: List of unique labels
        """
        link_content = {
            "file_link": file_linkage,
            "labels": labels,
        }

        link_yaml_path = self.output / "link.yaml"

        with open(link_yaml_path, "w") as f:
            yaml.dump(link_content, f, default_flow_style=False)
