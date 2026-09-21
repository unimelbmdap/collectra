"""
API backend for annotation display logic via pywebview's js_api interface.

JavaScript usage:
    await pywebview.api.load_yaml("path/to/results.yaml")
    const result = await pywebview.api.get_display_value("node_id")
    const nodes = await pywebview.api.get_all_nodes()
"""

import base64
import enum
import platform
from pathlib import Path
from typing import Annotated

import typer
import webview
import yaml
from rich import print

from collectra_gui.data_display import CollectraGraph, NodeDisplayValue

app = typer.Typer()


class ImageFormat(enum.Enum):
    PNG = "image/png"
    JPEG = "image/jpeg"
    JPG = "image/jpg"
    GIF = "image/gif"
    BMP = "image/bmp"
    WEBP = "image/webp"
    TIFF = "image/tiff"
    TIF = "image/tiff"


class GUIBackend:
    """
    API class exposed to JavaScript via pywebview.

    All public methods are accessible in JS as:
        window.pywebview.api.methodName(args)
    """

    def __init__(self, workflow: str = "", extension: str = ""):
        """
        Initialize the API instance.

        Sets up the annotation graph, YAML path, and window reference.
        All values are initially None until load_yaml() is called.
        """
        self._graph: CollectraGraph | None = None
        self._yaml_path: str | None = None
        self._window = None
        self._collectra_folders: list[dict] = []
        self._parent_folder: str | None = None
        self._global_labels: set[str] = set()
        self._global_label_counts: dict[str, int] = {}  # {label: total_count}
        self._extension = self._identify_extension(workflow, extension)
        self._workflow_dir: Path | None = Path(workflow) if workflow else None

    def _identify_extension(self, workflow: str, extension: str) -> str:
        workflow_path = Path(workflow)
        if workflow_path.is_dir() and (workflow_path / "pipeline.yaml").is_file():
            with open(workflow_path / "pipeline.yaml", "r") as f:
                config = yaml.safe_load(f)
            ext = config.get("collectra_pipeline_metadata", {}).get("ext", "")
            if ext:
                return ext
        if extension:
            return extension
        raise ValueError(
            "Either a valid workflow directory or file extension must be provided."
        )

    def set_window(self, window):
        """Store the window reference for use in dialogs."""
        self._window = window

    def _scan_collectra_folder(self, folder_path: Path) -> dict | None:
        """
        Scan a folder for YAML and image files.

        Args:
            folder_path: Path to the folder to scan

        Returns:
            dict with yaml_path and image_path if both found, None otherwise
        """
        yaml_path = None
        image_path = None
        image_extensions = [f".{ext.lower()}" for ext in ImageFormat.__members__.keys()]

        for filename in sorted(folder_path.iterdir(), reverse=True):
            if not filename.is_file():
                continue

            if filename.suffix in [".yaml", ".yml"] and yaml_path is None:
                yaml_path = filename

            if filename.suffix.lower() in image_extensions and image_path is None:
                image_path = filename

            if yaml_path and image_path:
                break

        if yaml_path is None or image_path is None:
            return None

        return {"yaml_path": str(yaml_path), "image_path": str(image_path)}

    def select_folder(self) -> dict:
        """
        Open native folder dialog and scan for YAML/image files.

        Returns:
            dict with folder_path, yaml_path, image_path (all absolute paths)
        """

        if self._window is None:
            return {"success": False, "error": "Window not initialized"}

        try:
            result = self._window.create_file_dialog(dialog_type=webview.FileDialog.FOLDER)

            if not result or len(result) == 0:
                return {"success": False, "error": "No folder selected"}

            folder_path = Path(result[0])

            yaml_path = None
            image_path = None
            image_extensions = [
                f".{ext.lower()}" for ext in ImageFormat.__members__.keys()
            ]

            for filename in folder_path.iterdir():

                if not filename.is_file():
                    continue

                if filename.suffix in [".yaml", ".yml"] and yaml_path is None:
                    yaml_path = filename

                if filename.suffix in image_extensions and image_path is None:
                    image_path = filename

                if yaml_path and image_path:
                    break

            if yaml_path is None:
                return {"success": False, "error": "No YAML file found in folder"}
            if image_path is None:
                return {"success": False, "error": "No image file found in folder"}

            return {
                "success": True,
                "folder_path": str(folder_path),
                "yaml_path": str(yaml_path),
                "image_path": str(image_path),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def select_parent_folder(self) -> dict:
        """
        Open native folder dialog and scan for subdirectories with the specified extension.

        Returns:
            dict with parent_path and list of folder names/indices
        """
        if self._window is None:
            return {"success": False, "error": "Window not initialized"}

        try:
            result = self._window.create_file_dialog(dialog_type=webview.FileDialog.FOLDER)

            if not result or len(result) == 0:
                return {"success": False, "error": "No folder selected"}

            parent_path = Path(result[0])
            return self._scan_parent_folder(parent_path)

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _detect_ext_near(self, path: Path) -> str:
        """Search path and its parents for a *.collectra folder and read ext from its pipeline.yaml."""
        for search_dir in [path, path.parent, path.parent.parent]:
            if not search_dir.is_dir():
                continue
            for entry in search_dir.iterdir():
                if entry.suffix == ".collectra" and (entry / "pipeline.yaml").exists():
                    with open(entry / "pipeline.yaml", "r") as f:
                        config = yaml.safe_load(f)
                    ext = config.get("collectra_pipeline_metadata", {}).get("ext", "")
                    if ext:
                        self._workflow_dir = entry
                        return ext
        return ""

    def select_collectra_path(self) -> dict:
        """
        Open native folder dialog and load either a single collectra folder
        (if it matches the extension) or a parent folder of collectra subfolders.

        Returns:
            dict with mode "single" or "parent" and the relevant data.
        """
        if self._window is None:
            return {"success": False, "error": "Window not initialized"}

        try:
            result = self._window.create_file_dialog(dialog_type=webview.FileDialog.FOLDER)

            if not result or len(result) == 0:
                return {"success": False, "error": "No folder selected"}

            selected_path = Path(result[0])

            # A previously-viewed page's yaml path is otherwise never cleared,
            # and _collectra_dirs() searches near it — so without this, asset
            # lookups (get_logo, get_theme_css) can silently keep matching a
            # file from whatever project was open before this new selection.
            self._yaml_path = None

            # Detect extension from the nearest *.collectra/pipeline.yaml
            detected_ext = self._detect_ext_near(selected_path)
            if detected_ext:
                self._extension = detected_ext

            if selected_path.suffix == f".{self._extension}":
                scan_result = self._scan_collectra_folder(selected_path)
                if scan_result is None:
                    return {
                        "success": False,
                        "error": "No YAML or image file found in folder",
                    }
                self._collectra_folders = []
                self._parent_folder = str(selected_path.parent)
                return {
                    "success": True,
                    "mode": "single",
                    "folder_path": str(selected_path),
                    "yaml_path": scan_result["yaml_path"],
                    "image_path": scan_result["image_path"],
                }

            parent_result = self._scan_parent_folder(selected_path)
            if parent_result.get("success"):
                parent_result["mode"] = "parent"
            return parent_result

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _scan_parent_folder(self, parent_path: Path) -> dict:
        self._parent_folder = str(parent_path)
        self._collectra_folders = []

        for entry in sorted(parent_path.iterdir()):
            if not entry.is_dir():
                continue
            if not entry.suffix == f".{self._extension}":
                continue

            scan_result = self._scan_collectra_folder(entry)
            if scan_result is not None:
                self._collectra_folders.append(
                    {
                        "name": entry.name,
                        "path": str(entry),
                        "yaml_path": scan_result["yaml_path"],
                        "image_path": scan_result["image_path"],
                    }
                )

        folders = [
            {"name": f["name"], "index": i}
            for i, f in enumerate(self._collectra_folders)
        ]

        # Compute aggregate label statistics across all folders
        self._global_label_counts = {}
        for folder in self._collectra_folders:
            try:
                with open(folder["yaml_path"], "r") as f:
                    yaml_data = yaml.safe_load(f)
                temp_graph = CollectraGraph.from_yaml_data(yaml_data)
                label_counts = temp_graph.count_nodes_by_label(
                    type_filter="collectra.ImageCrop"
                )
                # Accumulate unique labels
                self._global_labels.update(temp_graph.get_unique_labels())
                # Accumulate counts
                for label, count in label_counts.items():
                    self._global_label_counts[label] = (
                        self._global_label_counts.get(label, 0) + count
                    )
            except Exception:
                # Skip folders that can't be loaded
                pass

        return {
            "success": True,
            "parent_path": str(parent_path),
            "folders": folders,
            "global_label_counts": self._global_label_counts,
            "global_total": sum(self._global_label_counts.values()),
            "ext": self._extension,
        }

    def load_collectra_folder(self, index: int) -> dict:
        """
        Load a folder by its index from the previously scanned parent folder.

        Args:
            index: Index of the folder in the _collectra_folders list

        Returns:
            dict with folder_name, yaml_path, image_path
        """
        if not self._collectra_folders:
            return {
                "success": False,
                "error": "No folders loaded. Call select_parent_folder first.",
            }

        if index < 0 or index >= len(self._collectra_folders):
            return {"success": False, "error": f"Invalid folder index: {index}"}

        folder = self._collectra_folders[index]
        return {
            "success": True,
            "folder_name": folder["name"],
            "folder_path": folder["path"],
            "yaml_path": folder["yaml_path"],
            "image_path": folder["image_path"],
        }

    def get_active_node_ids(self, page_index: int) -> dict:
        """Return the labels, real types, and real result ids that belong to a page.

        `results.yaml` indexes each node by an internal id (a label plus a
        generated suffix for anything not an original source node), but the
        DAG built from pipeline.yaml uses the plain label as its node id.
        Correlating against the DAG must go through `label`, not `id`.

        The frontend greys out and disables every DAG node whose id is NOT in
        active_ids, can use node_types[label] to show the node's real type,
        and node_ids[label] to show the real results.yaml id(s) — a list,
        not a single value, since a detector-type node (e.g.
        collectra.ObjectDetectionYOLO) declares one output label in
        pipeline.yaml but can produce any number of real instances per page
        (one per detected object), all sharing that same label.
        """
        if not self._collectra_folders:
            return {"success": False, "error": "No folders loaded."}
        if page_index < 0 or page_index >= len(self._collectra_folders):
            return {"success": False, "error": f"Invalid page index: {page_index}"}

        folder = self._collectra_folders[page_index]
        try:
            with open(folder["yaml_path"], "r") as f:
                yaml_data = yaml.safe_load(f) or {}
            graph = CollectraGraph.from_yaml_data(yaml_data)
            node_types: dict[str, str] = {}
            node_ids: dict[str, list[str]] = {}
            for node_id in graph.nodes:
                node = graph.get_node(node_id)
                if node and node.label:
                    node_types[node.label] = node.type
                    node_ids.setdefault(node.label, []).append(node_id)
            return {
                "success": True,
                "active_ids": list(node_types.keys()),
                "node_types": node_types,
                "node_ids": node_ids,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_image_base64(self, image_path: str) -> dict:
        """
        Read an image file and return its base64 data.

        Args:
            image_path: Absolute path to the image file

        Returns:
            dict with base64-encoded data URI
        """
        try:
            extension = Path(image_path).suffix.lower()
            mime_type = ImageFormat[extension[1:].upper()].value

            with open(image_path, "rb") as f:
                image_data = f.read()

            base64_data = base64.b64encode(image_data).decode("utf-8")

            return {"success": True, "data": f"data:{mime_type};base64,{base64_data}"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def load_yaml(self, path: str) -> dict:
        """
        Load a YAML file and build the annotation graph.

        Args:
            path: Path to the YAML file

        Returns:
            dict with 'success', 'node_count', 'edge_count', or 'error'
        """
        try:
            with open(path, "r") as f:
                yaml_data = yaml.safe_load(f)

            self._graph = CollectraGraph.from_yaml_data(yaml_data)
            self._yaml_path = path

            # Accumulate labels from this graph
            self._global_labels.update(self._graph.get_unique_labels())

            return {"success": True, "path": path}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_display_value(self, node_id: str) -> dict:
        """
        Compute display value for an annotation.

        Args:
            node_id: The annotation ID

        Returns:
            dict with 'value', 'source_id', 'reason', or 'error'
        """
        if self._graph is None:
            return {"success": False, "error": "No graph loaded. Call load_yaml first."}

        try:
            result = self._graph.compute_display_value(self._graph.resolve_id(node_id))
            return {
                "success": True,
                "value": result.value,
                "source_id": result.source_id,
                "crop_region": result.crop_region,
                "reason": result.reason,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_page_texts(self) -> dict:
        """Return markdown and TEI content for the current page folder."""
        if self._yaml_path is None:
            return {"success": False, "error": "No page loaded."}
        page_dir = Path(self._yaml_path).parent
        result: dict = {"success": True}
        for key, filename in (("markdown", "markdown.md"), ("tei", "tei.xml")):
            p = page_dir / filename
            if p.exists():
                result[key] = p.read_text(encoding="utf-8")
                result[f"{key}_found"] = True
            else:
                result[key] = None
                result[f"{key}_found"] = False
        return result

    def update_page_text(self, kind: str, content: str) -> dict:
        """Write markdown or TEI content back to the current page's folder.

        Args:
            kind: "markdown" or "tei"
            content: the full text to write

        Returns:
            dict with 'success' or 'error'
        """
        if self._yaml_path is None:
            return {"success": False, "error": "No page loaded."}

        filenames = {"markdown": "markdown.md", "tei": "tei.xml"}
        filename = filenames.get(kind)
        if filename is None:
            return {"success": False, "error": f"Unknown kind: {kind}"}

        try:
            page_dir = Path(self._yaml_path).parent
            (page_dir / filename).write_text(content, encoding="utf-8")
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_all_nodes(self) -> dict:
        """
        Get all node IDs in the graph.

        Returns:
            dict with 'nodes' list or 'error'
        """
        if self._graph is None:
            return {"success": False, "error": "No graph loaded. Call load_yaml first."}

        return {"success": True, "nodes": self._graph.nodes}

    def get_node_info(self, node_id: str) -> dict:
        """
        Get detailed information about a node.

        Args:
            node_id: The annotation ID

        Returns:
            dict with node type, data, children, parents, or 'error'
        """
        if self._graph is None:
            return {"success": False, "error": "No graph loaded. Call load_yaml first."}

        node_id = self._graph.resolve_id(node_id)
        if node_id not in self._graph.nodes:
            return {"success": False, "error": f"Node '{node_id}' not found"}

        return {
            "success": True,
            "id": node_id,
            "type": self._graph.get_type(node_id),
            "data": self._graph.get_data(node_id),
            "children": self._graph.children(node_id),
            "parents": self._graph.parents(node_id),
        }

    def get_nodes_by_type(self, type_filter: str) -> dict:
        """
        Get all nodes containing a type substring.

        Args:
            type_filter: Substring to match in node types (e.g., "ImageCrop", "Text")

        Returns:
            dict with 'nodes' list of matching node IDs
        """
        if self._graph is None:
            return {"success": False, "error": "No graph loaded. Call load_yaml first."}

        matching = [
            node_id
            for node_id in self._graph.nodes
            if type_filter in self._graph.get_type(node_id)
        ]

        return {"success": True, "nodes": matching, "count": len(matching)}

    def get_available_labels(self) -> dict:
        """
        Get all unique labels accumulated across all loaded files.

        Returns:
            dict with 'success' and 'labels' list (sorted)
        """
        try:
            labels = sorted(self._global_labels)
            return {"success": True, "labels": labels}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_label_statistics(self) -> dict:
        """
        Return counts of nodes per label in current graph.

        Returns:
            dict with 'success', 'label_counts', and 'total'
        """
        if self._graph is None:
            return {"success": False, "error": "No graph loaded"}

        # Count only ImageCrop types (annotations visible in grid)
        label_counts = self._graph.count_nodes_by_label(
            type_filter="collectra.ImageCrop"
        )

        return {
            "success": True,
            "label_counts": label_counts,
            "total": sum(label_counts.values()),
        }

    def get_global_label_statistics(self) -> dict:
        """
        Return aggregated label counts across all files in parent folder.

        Returns:
            dict with 'success', 'label_counts', and 'total'
        """
        return {
            "success": True,
            "label_counts": self._global_label_counts,
            "total": sum(self._global_label_counts.values()),
        }

    def get_all_nodes_for_grid(self) -> dict:
        """
        Return all nodes formatted for AG Grid rows.

        Returns:
            dict with 'rows' list containing node data for grid display
        """
        if self._graph is None:
            return {"success": False, "error": "No graph loaded. Call load_yaml first."}

        rows = []
        for node_id in self._graph.nodes:
            node_display_value: NodeDisplayValue = self._graph.compute_display_value(
                node_id
            )
            rows.append(
                {
                    "id": node_id,
                    "type": self._graph.get_type(node_id),
                    "data": str(self._graph.get_data(node_id)),
                    "name": self._graph.get_name(node_id),
                    "label": self._graph.get_label(node_id),
                    "displayValue": node_display_value.value,
                    "crop_region": node_display_value.crop_region,
                    "displaySourceId": node_display_value.source_id,
                    "reason": node_display_value.reason,
                    "parents": ", ".join(self._graph.parents(node_id)),
                    "children": ", ".join(self._graph.children(node_id)),
                    "locked": node_display_value.locked,
                    "orientation": self._graph.get_orientation(node_id),
                }
            )

        return {"success": True, "rows": rows}

    def update_node_data(
        self, node_id: str, new_data: str, crop_id: str | None = None
    ) -> dict:
        """
        Update the data of a specific node.

        Args:
            node_id: The annotation ID
            new_data: New data to set for the node

        Returns:
            dict with 'success' or 'error'
        """
        try:
            if self._graph is None:
                raise ValueError("No graph loaded. Call load_yaml first.")
            # node_id may be "" when creating a new Text node via crop_id — only
            # resolve a real, non-empty id (an empty label would never match).
            resolved_id = self._graph.resolve_id(node_id) if node_id else node_id
            self._graph.set_data(resolved_id, new_data, crop_id)
            self._save_to_yaml()
            return self.get_all_nodes_for_grid()
        except Exception as e:
            return {"success": False, "error": str(e)}

    def update_node_coordinates(self, node_id: str, crop_region: dict) -> dict:
        """
        Update the crop region coordinates of a specific node.

        Args:
            node_id: The annotation ID
            crop_region: dict with x_center, y_center, width_relative, height_relative

        Returns:
            dict with 'success' and updated rows, or 'error'
        """
        try:
            if self._graph is None:
                raise ValueError("No graph loaded. Call load_yaml first.")
            self._graph.set_crop_region(self._graph.resolve_id(node_id), crop_region)
            self._save_to_yaml()
            return self.get_all_nodes_for_grid()
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _next_annotation_id(self, label: str) -> str:
        """Matches the pipeline's scheme (label+N) — reuses the smallest
        suffix a deletion freed up, rather than always growing past the
        highest one ever used. Shared with preview_annotation_id."""
        existing_numbers = set()
        for node_id in self._graph.nodes:
            node = self._graph.get_node(node_id)
            if node and node.label == label:
                suffix = node_id[len(label):]
                if suffix.isdigit():
                    existing_numbers.add(int(suffix))
        n = 1
        while n in existing_numbers:
            n += 1
        return f"{label}{n}"

    def preview_annotation_id(self, label: str) -> dict:
        """What create_annotation would name the next box with this label, without creating anything."""
        if self._graph is None:
            return {"success": False, "error": "No graph loaded. Call load_yaml first."}
        return {"success": True, "id": self._next_annotation_id(label)}

    def create_annotation(
        self,
        crop_region: dict,
        label: str,
        parent_id: str,
        image_path: str = "",
        name: str = "",
        crop_id: str = "",
    ) -> dict:
        """
        Create a new ImageCrop annotation with a Text child.

        Args:
            crop_region: dict with x_center, y_center, width_relative, height_relative
            name: optional per-instance display name, distinct from label
            crop_id: explicit node id; auto-numbered (label + N) when blank

        Returns:
            dict with updated grid data or 'error'
        """
        if self._graph is None:
            return {"success": False, "error": "No graph loaded. Call load_yaml first."}

        if not image_path:
            return {
                "success": False,
                "error": "Image path is required to create annotation.",
            }

        try:
            # Find root Image node
            root_image_id = None
            for node_id in self._graph.nodes:
                node_type = self._graph.get_type(node_id)
                if "Image" in node_type and "ImageCrop" not in node_type:
                    root_image_id = node_id
                    break

            if root_image_id is None:
                return {"success": False, "error": "No root Image node found in graph."}

            orientation = "north"

            if parent_id:
                parent_node = self._graph.get_node(parent_id)
                if parent_node:
                    orientation = parent_node.orientation

            crop_id = crop_id.strip() or self._next_annotation_id(label)
            if crop_id in self._graph.nodes:
                return {"success": False, "error": f"Id '{crop_id}' already exists."}

            if self._yaml_path is None:
                raise ValueError("YAML path is not set.")

            # Create ImageCrop node
            crop_data = {
                "label": label,
                "type": "collectra.ImageCrop",
                "id": crop_id,
                "parents": parent_id if parent_id else root_image_id,
                "orientation": orientation,
                "data": Path(image_path).name,
                "name": name,
                **crop_region,
            }
            self._graph.add_node(crop_data)
            self._global_labels.add(label)
            # Update global label counts if in parent folder mode
            if self._collectra_folders:
                self._global_label_counts[label] = (
                    self._global_label_counts.get(label, 0) + 1
                )
            # Save and return updated grid
            self._save_to_yaml()
            result = self.get_all_nodes_for_grid()
            # Include updated global stats in response
            if self._collectra_folders:
                result["global_label_counts"] = self._global_label_counts
                result["global_total"] = sum(self._global_label_counts.values())
            return result

        except Exception as e:
            return {"success": False, "error": str(e)}

    def delete_annotation(self, node_id: str) -> dict:
        """
        Delete an ImageCrop annotation and everything nested under it —
        cascades to its Text children, any child ImageCrops (sub-crops), and
        so on at any depth. Deleting a crop that still has children used to
        leave them behind pointing at a now-nonexistent parent id (a dangling
        reference the graph library then treats as a bare placeholder node,
        breaking id-uniqueness checks for that id ever after) — cascading is
        the only way to guarantee no orphans.

        Args:
            node_id: The ImageCrop annotation ID to delete

        Returns:
            dict with updated grid data on success, or 'error' on failure
        """
        if self._graph is None:
            return {"success": False, "error": "No graph loaded. Call load_yaml first."}

        try:
            node_id = self._graph.resolve_id(node_id)

            # Collect every descendant (any type, any depth) before deleting
            # anything — walking the graph while removing nodes from it would
            # skip some.
            to_delete = []
            queue = [node_id]
            seen = {node_id}
            while queue:
                current = queue.pop(0)
                to_delete.append(current)
                for child_id in self._graph.children(current):
                    if child_id not in seen:
                        seen.add(child_id)
                        queue.append(child_id)

            deleted_labels = [
                node.label
                for descendant_id in to_delete
                if (node := self._graph.get_node(descendant_id))
            ]

            # Deepest-reached first, so a node's own children are already
            # gone before it is (remove_node cleans up edges regardless, but
            # this mirrors "delete what's inside first").
            for descendant_id in reversed(to_delete):
                self._graph.remove_node(descendant_id)

            # Update global label counts if in parent folder mode
            if self._collectra_folders:
                for label in deleted_labels:
                    if label in self._global_label_counts:
                        self._global_label_counts[label] -= 1
                        if self._global_label_counts[label] <= 0:
                            del self._global_label_counts[label]

            # Save and return updated grid
            self._save_to_yaml()
            result = self.get_all_nodes_for_grid()
            # Every id actually removed (the requested one plus every
            # cascaded descendant) — the frontend needs this to close any
            # tab left open on a descendant that wasn't the one you directly
            # deleted, not just the top-level id.
            result["deleted_ids"] = to_delete
            # Include updated global stats in response
            if self._collectra_folders:
                result["global_label_counts"] = self._global_label_counts
                result["global_total"] = sum(self._global_label_counts.values())
            return result

        except Exception as e:
            return {"success": False, "error": str(e)}

    def rename_annotation(
        self,
        node_id: str,
        name: str,
        label: str | None = None,
        new_id: str | None = None,
    ) -> dict:
        """
        Sets an ImageCrop annotation's display name. label and new_id are
        optional; omitted, that field is left as-is.

        Args:
            node_id: The ImageCrop annotation ID to rename
            name: The new name
            label: optional new label
            new_id: optional new id — re-points every parent reference to match

        Returns:
            dict with updated grid data on success, or 'error' on failure
        """
        if self._graph is None:
            return {"success": False, "error": "No graph loaded. Call load_yaml first."}

        try:
            node_id = self._graph.resolve_id(node_id)
            if new_id and new_id != node_id:
                self._graph.rename_id(node_id, new_id)
                node_id = new_id
            if label is not None:
                self._graph.set_label(node_id, label)
            self._graph.set_name(node_id, name)
            self._save_to_yaml()
            return self.get_all_nodes_for_grid()

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _save_to_yaml(self) -> None:
        """Save the current graph back to the original YAML file."""
        if self._yaml_path is None or self._graph is None:
            raise ValueError("No YAML file loaded to save to.")
        yaml_data = self._graph.to_yaml_data()
        with open(self._yaml_path, "w") as f:
            for key, value in yaml_data.items():
                yaml.dump(
                    {key: value if not isinstance(value, list) else value},
                    f,
                    sort_keys=False,
                )
                f.write("\n")

    def _collectra_dirs(self) -> list[Path]:
        """Return all .collectra directories to search, in priority order.

        Search order: workflow_dir → root itself → root's children → root's parent's children.
        The parent search lets the user open a book sub-folder (e.g. Livy-Books21-25/)
        while the .collectra lives as a sibling inside the project root (LivyProject/).
        """
        dirs: list[Path] = []
        if self._workflow_dir and self._workflow_dir.is_dir():
            dirs.append(self._workflow_dir)
        roots: list[Path] = []
        if self._parent_folder:
            roots.append(Path(self._parent_folder))
        if self._yaml_path:
            roots.append(Path(self._yaml_path).parent.parent)
        seen = set()

        def _scan(directory: Path) -> None:
            """Add any .collectra dirs found directly inside `directory`."""
            if not directory.is_dir():
                return
            if directory.suffix == ".collectra" and directory not in seen:
                dirs.append(directory)
                seen.add(directory)
                return
            try:
                for entry in directory.iterdir():
                    if entry.is_dir() and entry.suffix == ".collectra" and entry not in seen:
                        dirs.append(entry)
                        seen.add(entry)
            except (PermissionError, OSError):
                pass

        for root in roots:
            _scan(root)           # look inside the selected folder
            _scan(root.parent)    # look inside the parent (sibling .collectra case)

        return dirs

    def _read_pipeline_metadata(self) -> dict:
        """Return the collectra_pipeline_metadata block from pipeline.yaml, or {} if absent."""
        for directory in self._collectra_dirs():
            try:
                matches = list(directory.rglob("pipeline.yaml"))
            except (PermissionError, OSError):
                continue
            if matches:
                try:
                    with open(matches[0], "r") as f:
                        data = yaml.safe_load(f) or {}
                    return data.get("collectra_pipeline_metadata", {}) or {}
                except Exception:
                    return {}
        return {}

    def get_pipeline_yaml(self) -> dict:
        """Read pipeline.yaml from the root of the .collectra directory."""
        for directory in self._collectra_dirs():
            pipeline_path = directory / "pipeline.yaml"
            if pipeline_path.is_file():
                try:
                    with open(pipeline_path, "r") as f:
                        return {"success": True, "yaml_text": f.read()}
                except Exception as e:
                    return {"success": False, "error": str(e)}
        return {"success": False, "error": "No pipeline.yaml found"}

    def get_theme_css(self) -> dict:
        """Load the CSS theme named in pipeline metadata, anywhere in the .collectra tree.

        Falls back to a file literally named theme.css, then any CSS file, when no
        named file is found.
        """
        named = self._read_pipeline_metadata().get("theme", "")
        for directory in self._collectra_dirs():
            # 1. Exact name from metadata
            if named:
                try:
                    hits = list(directory.rglob(named))
                except (PermissionError, OSError):
                    hits = []
                if hits:
                    try:
                        with open(hits[0], "r") as f:
                            return {"success": True, "css": f.read(), "path": str(hits[0])}
                    except Exception as e:
                        return {"success": False, "error": str(e)}

            # 2. Fallback: prefer a file named theme.css, else any CSS file
            try:
                css_files = [f for f in directory.iterdir() if f.suffix == ".css" and f.is_file()]
            except (PermissionError, OSError):
                continue
            preferred = next((f for f in css_files if f.stem == "theme"), None)
            target = preferred or (css_files[0] if css_files else None)
            if target:
                try:
                    with open(target, "r") as f:
                        return {"success": True, "css": f.read(), "path": str(target)}
                except Exception as e:
                    return {"success": False, "error": str(e)}

        # Fall back to bundled default asset
        asset_path = Path(get_resource_path("assets/theme.css"))
        if asset_path.is_file():
            try:
                with open(asset_path, "r") as f:
                    return {"success": True, "css": f.read(), "path": str(asset_path)}
            except Exception as e:
                return {"success": False, "error": str(e)}

        return {"success": False, "error": "No CSS theme file found"}

    def _encode_logo(self, path: Path) -> dict:
        """Encode a logo file (PNG/SVG/etc.) as a base64 data URI with the right MIME type."""
        try:
            ext = path.suffix.lower()
            if ext == ".svg":
                mime = "image/svg+xml"
            else:
                mime = ImageFormat[ext[1:].upper()].value
            with open(path, "rb") as f:
                data = base64.b64encode(f.read()).decode("utf-8")
            return {"success": True, "data": f"data:{mime};base64,{data}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_logo(self) -> dict:
        """Find the logo named in pipeline metadata (PNG or SVG), anywhere in the .collectra tree.

        Falls back to globbing for a logo-ish image when no named file is found,
        then to Collectra's own bundled logo when the project has none at all —
        callers rely on this always succeeding so a project with no logo doesn't
        keep showing whichever project's logo loaded previously.
        """
        named = self._read_pipeline_metadata().get("logo", "")
        image_exts = {f".{e.lower()}" for e in ImageFormat.__members__} | {".svg"}
        for directory in self._collectra_dirs():
            # 1. Exact name from metadata
            if named:
                try:
                    hits = list(directory.rglob(named))
                except (PermissionError, OSError):
                    hits = []
                if hits:
                    return self._encode_logo(hits[0])
            # 2. Glob fallback: prefer a "logo"-named image, else any image/SVG
            try:
                candidates = [
                    f for f in directory.rglob("*") if f.suffix.lower() in image_exts
                ]
            except (PermissionError, OSError):
                continue
            logos = [f for f in candidates if "logo" in f.stem.lower()]
            target = logos[0] if logos else (candidates[0] if candidates else None)
            if target:
                return self._encode_logo(target)

        # Fall back to bundled default asset
        asset_path = Path(get_resource_path("assets/CollectraLogoAndTitle.svg"))
        if asset_path.is_file():
            return self._encode_logo(asset_path)

        return {"success": False, "error": "No logo file found"}

def _disable_macos_tabbing() -> None:
    """Disable macOS Window Tab Bar for the pywebview window."""
    import platform
    if platform.system() != "Darwin":
        return
    try:
        from AppKit import NSApplication
        for win in NSApplication.sharedApplication().windows():
            win.setTabbingMode_(2)  # NSWindowTabbingModeDisallowed = 2
    except Exception:
        pass


def get_resource_path(relative_path: str) -> str:
    """Get the path to a bundled resource, works for dev and PyInstaller."""
    import sys

    if getattr(sys, "frozen", False):
        # Running as PyInstaller bundle
        base_path = Path(getattr(sys, "_MEIPASS", ""))
    else:
        # Running in development
        base_path = Path(__file__).parent
    return str(base_path / relative_path)


@app.command()
def start(
    workflow: Annotated[
        str, typer.Option("--workflow", "-w", help="Specify the workflow to use")
    ] = "",
    ext: Annotated[
        str,
        typer.Option("--extension", "-e", help="Specify the file extension to filter"),
    ] = "",
    debug: bool = False,
):
    """
    Create and start the pywebview window with the API.

    Args:
        debug: Enable developer tools
    """
    if not (workflow or ext):
        print(
            "[red]Error:[/red] --workflow or --extension arguments are required to start the GUI."
        )
        raise typer.Exit(code=1)
    api = GUIBackend(workflow=workflow, extension=ext)
    html_path = get_resource_path("index.html")
    window = webview.create_window(
        title="Collectra Viewer", url=html_path, js_api=api, width=1200, height=800,
        min_size=(800, 600)
    )

    def on_started():
        api.set_window(window)
        _disable_macos_tabbing()

    webview.start(func=on_started, debug=debug)
    return window


if __name__ == "__main__":
    app()
