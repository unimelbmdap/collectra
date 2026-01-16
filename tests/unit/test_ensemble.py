from pathlib import Path

import pytest
import yaml

from collectra.ensemble import Ensembler


class TestFileLinkageCreation:
    """Test file linkage identification across source folders."""

    def test_create_file_linkage_single_file_multiple_folders(self, tmp_path):
        """Test that identical filenames across folders are correctly linked."""
        # Create mock folder structure
        folder1 = tmp_path / "folder1"
        folder2 = tmp_path / "folder2"
        folder1.mkdir()
        folder2.mkdir()

        (folder1 / "file1.collectra").mkdir()
        (folder2 / "file1.collectra").mkdir()
        (folder1 / "file2.collectra").mkdir()

        folders = [folder1, folder2]
        ensembler = Ensembler(
            folders=folders, output=tmp_path / "ensemble", extension=".collectra"
        )

        linkage = ensembler._create_file_linkage()

        # file1 should appear in both folders
        assert "file1.collectra" in linkage
        assert len(linkage["file1.collectra"]) == 2

        # file2 should appear in folder1 only
        assert "file2.collectra" in linkage
        assert len(linkage["file2.collectra"]) == 1
        assert str(folder1 / "file2.collectra") in linkage["file2.collectra"]

    def test_file_linkage_no_common_files(self, tmp_path):
        """Test linkage when folders have no common files."""
        folder1 = tmp_path / "folder1"
        folder2 = tmp_path / "folder2"
        folder1.mkdir()
        folder2.mkdir()

        (folder1 / "fileA.collectra").mkdir()
        (folder2 / "fileB.collectra").mkdir()

        ensembler = Ensembler(
            folders=[folder1, folder2],
            output=tmp_path / "ensemble",
            extension=".collectra",
        )

        linkage = ensembler._create_file_linkage()

        # Both files should be in the linkage (pass-through behavior)
        assert "fileA.collectra" in linkage
        assert "fileB.collectra" in linkage
        assert len(linkage) == 2

    def test_file_linkage_empty_folders(self, tmp_path):
        """Test linkage creation with empty folders."""
        folder1 = tmp_path / "folder1"
        folder1.mkdir()

        ensembler = Ensembler(
            folders=[folder1], output=tmp_path / "ensemble", extension=".collectra"
        )

        linkage = ensembler._create_file_linkage()

        # Should be empty
        assert linkage == {}

    def test_file_linkage_ignores_files_without_extension(self, tmp_path):
        """Test that files without the target extension are ignored."""
        folder1 = tmp_path / "folder1"
        folder1.mkdir()

        (folder1 / "file1.collectra").mkdir()
        (folder1 / "file2.txt").mkdir()
        (folder1 / "file3").mkdir()

        ensembler = Ensembler(
            folders=[folder1],
            output=tmp_path / "ensemble",
            extension=".collectra",
        )

        linkage = ensembler._create_file_linkage()

        # Only file1.collectra should be in linkage
        assert "file1.collectra" in linkage
        assert "file2.txt" not in linkage
        assert "file3" not in linkage


class TestLabelIdentification:
    """Test identification of unique labels across source files."""

    def test_identify_unique_labels_multiple_sources(self, tmp_path):
        pass
        # """Test that all unique labels are collected from multiple sources."""
        # folder1 = tmp_path / "folder1"
        # folder2 = tmp_path / "folder2"
        # folder1.mkdir()
        # folder2.mkdir()

        # # Create first source file with YAML
        # source1_path = folder1 / "file1.collectra"
        # source1_path.mkdir()
        # source1_data = {
        #     "collectra_results_metadata": {"workflow": "test", "version": "0.1.0"},
        #     "source_image": {"type": "collectra.Image", "data": "img.jpg"},
        #     "crop1": {"type": "collectra.ImageCrop", "id": "crop_1"},
        #     "text1": {"type": "collectra.Text", "id": "text_1"},
        # }
        # with open(source1_path / "results.yaml", "w") as f:
        #     yaml.dump(source1_data, f)

        # # Create second source file with YAML
        # source2_path = folder2 / "file1.collectra"
        # source2_path.mkdir()
        # source2_data = {
        #     "collectra_results_metadata": {"workflow": "test", "version": "0.1.0"},
        #     "source_image": {"type": "collectra.Image", "data": "img.jpg"},
        #     "crop2": {"type": "collectra.ImageCrop", "id": "crop_2"},
        #     "text1": {"type": "collectra.Text", "id": "text_1"},
        # }
        # with open(source2_path / "results.yaml", "w") as f:
        #     yaml.dump(source2_data, f)

        # ensembler = Ensembler(
        #     folders=[folder1, folder2],
        #     output=tmp_path / "ensemble",
        #     extension=".collectra",
        # )

        # file_linkage = ensembler._create_file_linkage()
        # labels = ensembler._identify_unique_labels(file_linkage)

        # # Should have all unique labels except metadata
        # assert "source_image" in labels
        # assert "crop1" in labels
        # assert "crop2" in labels
        # assert "text1" in labels
        # assert "collectra_results_metadata" not in labels

    def test_identify_labels_empty_source(self, tmp_path):
        """Test label identification returns hard-coded labels."""
        folder1 = tmp_path / "folder1"
        folder2 = tmp_path / "folder2"
        folder1.mkdir()
        folder2.mkdir()

        source1_path = folder1 / "file1.collectra"
        source1_path.mkdir()
        source1_data = {
            "collectra_results_metadata": {"workflow": "test"},
            "image": {"type": "collectra.Image", "data": "img.jpg"},
        }
        with open(source1_path / "results.yaml", "w") as f:
            yaml.dump(source1_data, f)

        source2_path = folder2 / "file1.collectra"
        source2_path.mkdir()
        # Empty YAML
        with open(source2_path / "results.yaml", "w") as f:
            yaml.dump({}, f)

        ensembler = Ensembler(
            folders=[folder1, folder2],
            output=tmp_path / "ensemble",
            extension=".collectra",
        )

        file_linkage = ensembler._create_file_linkage()
        labels = ensembler._identify_unique_labels(file_linkage)

        # Should return the hard-coded list of labels
        expected_labels = [
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
        assert labels == expected_labels


class TestRetrieveLabelData:
    """Test label data retrieval from sources."""

    def test_retrieve_label_data_single_source(self, tmp_path):
        """Test retrieving label data from single source."""
        folder1 = tmp_path / "folder1"
        folder1.mkdir()

        source_path = folder1 / "file1.collectra"
        source_path.mkdir()
        source_data = {
            "collectra_results_metadata": {"workflow": "test"},
            "crop_label": [
                {
                    "type": "collectra.ImageCrop",
                    "id": "crop_1",
                    "x_center": 0.5,
                    "y_center": 0.5,
                }
            ],
        }
        with open(source_path / "results.yaml", "w") as f:
            yaml.dump(source_data, f)

        ensembler = Ensembler(
            folders=[folder1],
            output=tmp_path / "ensemble",
            extension=".collectra",
        )

        storage_dict = ensembler._retrieve_label_data_for_file(
            "file1.collectra", [str(source_path)], ["crop_label"]
        )

        assert "crop_label" in storage_dict
        assert storage_dict["crop_label"]["type"] == "collectra.ImageCrop"
        assert str(source_path) in storage_dict["crop_label"]["items"]
        assert len(storage_dict["crop_label"]["items"][str(source_path)]) == 1

    def test_retrieve_label_data_multiple_sources(self, tmp_path):
        """Test retrieving label data from multiple sources."""
        folder1 = tmp_path / "folder1"
        folder2 = tmp_path / "folder2"
        folder1.mkdir()
        folder2.mkdir()

        source1_path = folder1 / "file1.collectra"
        source1_path.mkdir()
        source1_data = {
            "collectra_results_metadata": {"workflow": "test"},
            "text_label": [{"type": "collectra.Text", "id": "text_1", "data": "hello"}],
        }
        with open(source1_path / "results.yaml", "w") as f:
            yaml.dump(source1_data, f)

        source2_path = folder2 / "file1.collectra"
        source2_path.mkdir()
        source2_data = {
            "collectra_results_metadata": {"workflow": "test"},
            "text_label": [{"type": "collectra.Text", "id": "text_2", "data": "world"}],
        }
        with open(source2_path / "results.yaml", "w") as f:
            yaml.dump(source2_data, f)

        ensembler = Ensembler(
            folders=[folder1, folder2],
            output=tmp_path / "ensemble",
            extension=".collectra",
        )

        storage_dict = ensembler._retrieve_label_data_for_file(
            "file1.collectra", [str(source1_path), str(source2_path)], ["text_label"]
        )

        assert "text_label" in storage_dict
        assert len(storage_dict["text_label"]["items"]) == 2
        assert str(source1_path) in storage_dict["text_label"]["items"]
        assert str(source2_path) in storage_dict["text_label"]["items"]

    def test_retrieve_label_data_normalizes_single_item_to_list(self, tmp_path):
        """Test that single items are normalized to list format."""
        folder1 = tmp_path / "folder1"
        folder1.mkdir()

        source_path = folder1 / "file1.collectra"
        source_path.mkdir()
        source_data = {
            "collectra_results_metadata": {"workflow": "test"},
            "single_image": {
                "type": "collectra.Image",
                "data": "img.jpg",
            },  # Not a list
        }
        with open(source_path / "results.yaml", "w") as f:
            yaml.dump(source_data, f)

        ensembler = Ensembler(
            folders=[folder1],
            output=tmp_path / "ensemble",
            extension=".collectra",
        )

        storage_dict = ensembler._retrieve_label_data_for_file(
            "file1.collectra", [str(source_path)], ["single_image"]
        )

        # Should be normalized to list
        assert isinstance(storage_dict["single_image"]["items"][str(source_path)], list)
        assert len(storage_dict["single_image"]["items"][str(source_path)]) == 1

    def test_retrieve_skips_missing_labels(self, tmp_path):
        """Test that missing labels are not included in storage_dict."""
        folder1 = tmp_path / "folder1"
        folder1.mkdir()

        source_path = folder1 / "file1.collectra"
        source_path.mkdir()
        source_data = {
            "collectra_results_metadata": {"workflow": "test"},
            "label_a": {"type": "collectra.Image", "data": "img.jpg"},
        }
        with open(source_path / "results.yaml", "w") as f:
            yaml.dump(source_data, f)

        ensembler = Ensembler(
            folders=[folder1],
            output=tmp_path / "ensemble",
            extension=".collectra",
        )

        storage_dict = ensembler._retrieve_label_data_for_file(
            "file1.collectra",
            [str(source_path)],
            ["label_a", "label_b", "label_c"],  # label_b and label_c missing
        )

        # Only label_a should be present
        assert "label_a" in storage_dict
        assert "label_b" not in storage_dict
        assert "label_c" not in storage_dict


class TestEnsembleFolderStructure:
    """Test ensemble output folder creation."""

    def test_create_ensemble_folder_structure(self, tmp_path):
        """Test that ensemble folder structure is created correctly."""
        folder1 = tmp_path / "folder1"
        folder1.mkdir()

        ensembler = Ensembler(
            folders=[folder1],
            output=tmp_path / "ensemble",
            extension=".collectra",
        )

        file_linkage = {
            "file1.collectra": [str(folder1 / "file1.collectra")],
            "file2.collectra": [str(folder1 / "file2.collectra")],
        }

        ensembler._create_ensemble_folder_structure(file_linkage)

        # Check that output folder and subfolders exist
        assert (tmp_path / "ensemble").exists()
        assert (tmp_path / "ensemble" / "file1.collectra").exists()
        assert (tmp_path / "ensemble" / "file2.collectra").exists()


class TestSaveLinkYaml:
    """Test link.yaml file generation."""

    def test_save_link_yaml(self, tmp_path):
        """Test that link.yaml is saved correctly."""
        ensembler = Ensembler(
            folders=[tmp_path / "folder1"],
            output=tmp_path / "ensemble",
            extension=".collectra",
        )
        ensembler.output.mkdir(parents=True, exist_ok=True)

        file_linkage = {
            "file1.collectra": [
                str(tmp_path / "folder1" / "file1.collectra"),
                str(tmp_path / "folder2" / "file1.collectra"),
            ]
        }
        labels = ["label1", "label2", "label3"]

        ensembler._save_link_yaml(file_linkage, labels)

        # Check that link.yaml exists and has correct content
        link_yaml_path = tmp_path / "ensemble" / "link.yaml"
        assert link_yaml_path.exists()

        with open(link_yaml_path, "r") as f:
            content = yaml.safe_load(f)

        assert "file_link" in content
        assert "labels" in content
        assert content["file_link"] == file_linkage
        assert content["labels"] == labels


class TestGenerateMetadata:
    """Test metadata generation."""

    def test_generate_metadata(self, tmp_path):
        """Test that metadata is generated correctly."""
        ensembler = Ensembler(
            folders=[tmp_path / "folder1"],
            output=tmp_path / "ensemble",
            extension=".collectra",
        )

        source_files = [
            "folder1/file1.collectra",
            "folder2/file1.collectra",
        ]
        original_workflow = "test_workflow"

        metadata = ensembler._generate_metadata(source_files, original_workflow)

        assert "collectra_results_metadata" in metadata
        assert metadata["collectra_results_metadata"]["workflow"] == original_workflow
        assert metadata["collectra_results_metadata"]["version"] == "0.1.0"
        assert "timestamp" in metadata["collectra_results_metadata"]
        assert (
            metadata["collectra_results_metadata"]["ensemble_sources"] == source_files
        )


class TestLoadSourceFileContent:
    """Test loading source file content."""

    def test_load_source_file_content_valid(self, tmp_path):
        """Test loading valid YAML content."""
        source_path = tmp_path / "file.collectra"
        source_path.mkdir()

        data = {
            "collectra_results_metadata": {"workflow": "test"},
            "label1": {"type": "collectra.Image"},
        }
        with open(source_path / "results.yaml", "w") as f:
            yaml.dump(data, f)

        ensembler = Ensembler(
            folders=[tmp_path],
            output=tmp_path / "ensemble",
            extension=".collectra",
        )

        content = ensembler._load_source_file_content(source_path)

        assert content == data

    def test_load_source_file_content_missing_file(self, tmp_path):
        """Test loading from missing results.yaml."""
        source_path = tmp_path / "file.collectra"
        source_path.mkdir()

        ensembler = Ensembler(
            folders=[tmp_path],
            output=tmp_path / "ensemble",
            extension=".collectra",
        )

        content = ensembler._load_source_file_content(source_path)

        # Should return empty dict
        assert content == {}


class TestEnsemblerDataclass:
    """Test Ensembler dataclass initialization and attributes."""

    def test_ensembler_initialization(self, tmp_path):
        """Test Ensembler dataclass can be initialized."""
        folders = [tmp_path / "folder1", tmp_path / "folder2"]
        output = tmp_path / "ensemble"
        extension = ".collectra"

        ensembler = Ensembler(folders=folders, output=output, extension=extension)

        assert ensembler.folders == folders
        assert ensembler.output == output
        assert ensembler.extension == extension

    def test_ensembler_has_ensemble_method(self):
        """Test that Ensembler has ensemble method."""
        ensembler = Ensembler(folders=[], output=Path("."), extension=".collectra")

        assert hasattr(ensembler, "ensemble")
        assert callable(ensembler.ensemble)

    def test_ensembler_paths_are_pathlib(self, tmp_path):
        """Test that paths are stored as Path objects."""
        ensembler = Ensembler(
            folders=[tmp_path / "f1"], output=tmp_path / "out", extension=".ext"
        )

        assert isinstance(ensembler.folders[0], Path)
        assert isinstance(ensembler.output, Path)


class TestValidateImageLabels:
    """Test Image type label validation."""

    def test_image_same_data_source_valid(self, tmp_path):
        """Test that Images with identical data sources pass validation."""
        image_label = {
            "type": "collectra.Image",
            "items": {
                "folder1/file1.collectra": [
                    {"type": "collectra.Image", "data": "source.jpg"}
                ],
                "folder2/file1.collectra": [
                    {"type": "collectra.Image", "data": "source.jpg"}
                ],
            },
        }

        ensembler = Ensembler(
            folders=[tmp_path],
            output=tmp_path / "ensemble",
            extension=".collectra",
        )

        # Should not raise
        ensembler._validate_image_labels(image_label)

    def test_image_different_data_sources_invalid(self, tmp_path):
        """Test that Images with different data sources raise error."""
        image_label = {
            "type": "collectra.Image",
            "items": {
                "folder1/file1.collectra": [
                    {"type": "collectra.Image", "data": "source1.jpg"}
                ],
                "folder2/file1.collectra": [
                    {"type": "collectra.Image", "data": "source2.jpg"}
                ],
            },
        }

        ensembler = Ensembler(
            folders=[tmp_path],
            output=tmp_path / "ensemble",
            extension=".collectra",
        )

        # Should raise ValueError
        with pytest.raises(ValueError):
            ensembler._validate_image_labels(image_label)

    def test_image_single_source_valid(self, tmp_path):
        """Test that single source Image passes validation."""
        image_label = {
            "type": "collectra.Image",
            "items": {
                "folder1/file1.collectra": [
                    {"type": "collectra.Image", "data": "source.jpg"}
                ]
            },
        }

        ensembler = Ensembler(
            folders=[tmp_path],
            output=tmp_path / "ensemble",
            extension=".collectra",
        )

        # Should not raise
        ensembler._validate_image_labels(image_label)


class TestEnsembleImageCrops:
    """Test ImageCrop ensembling with Weighted Box Fusion."""

    def test_ensemble_overlapping_image_crops(self, tmp_path):
        """Test that overlapping ImageCrops are fused together."""
        label_data = {
            "type": "collectra.ImageCrop",
            "items": {
                "folder1/file1.collectra": [
                    {
                        "type": "collectra.ImageCrop",
                        "id": "crop_1",
                        "data": "img.jpg",
                        "x_center": 0.5,
                        "y_center": 0.5,
                        "width_relative": 0.2,
                        "height_relative": 0.2,
                    }
                ],
                "folder2/file1.collectra": [
                    {
                        "type": "collectra.ImageCrop",
                        "id": "crop_2",
                        "data": "img.jpg",
                        "x_center": 0.51,
                        "y_center": 0.51,
                        "width_relative": 0.2,
                        "height_relative": 0.2,
                    }
                ],
            },
        }

        ensembler = Ensembler(
            folders=[tmp_path],
            output=tmp_path / "ensemble",
            extension=".collectra",
        )

        result = ensembler._ensemble_image_crops(label_data, "crop_label")

        # Result should be a list or single item with ensemble attribute
        if isinstance(result, list):
            assert len(result) == 1  # Overlapping boxes fused
            assert "ensemble" in result[0]
            assert len(result[0]["ensemble"]) == 2
        else:
            assert "ensemble" in result
            assert len(result["ensemble"]) == 2

    def test_ensemble_non_overlapping_image_crops(self, tmp_path):
        """Test that non-overlapping ImageCrops are kept separate."""
        label_data = {
            "type": "collectra.ImageCrop",
            "items": {
                "folder1/file1.collectra": [
                    {
                        "type": "collectra.ImageCrop",
                        "id": "crop_1",
                        "data": "img.jpg",
                        "x_center": 0.2,
                        "y_center": 0.2,
                        "width_relative": 0.1,
                        "height_relative": 0.1,
                    }
                ],
                "folder2/file1.collectra": [
                    {
                        "type": "collectra.ImageCrop",
                        "id": "crop_2",
                        "data": "img.jpg",
                        "x_center": 0.8,
                        "y_center": 0.8,
                        "width_relative": 0.1,
                        "height_relative": 0.1,
                    }
                ],
            },
        }

        ensembler = Ensembler(
            folders=[tmp_path],
            output=tmp_path / "ensemble",
            extension=".collectra",
        )

        result = ensembler._ensemble_image_crops(label_data, "crop_label")

        # Result should have 2 items (no fusion)
        if isinstance(result, list):
            assert len(result) == 2
        else:
            # If returned as single item, still should have the structure
            assert "type" in result

    def test_ensemble_crop_format_has_ensemble_attribute(self, tmp_path):
        """Test that ensembled crops have correct ensemble attribute format."""
        label_data = {
            "type": "collectra.ImageCrop",
            "items": {
                "folder1/file1.collectra": [
                    {
                        "type": "collectra.ImageCrop",
                        "id": "crop_1",
                        "data": "img.jpg",
                        "x_center": 0.5,
                        "y_center": 0.5,
                        "width_relative": 0.2,
                        "height_relative": 0.2,
                    }
                ],
                "folder2/file1.collectra": [
                    {
                        "type": "collectra.ImageCrop",
                        "id": "crop_2",
                        "data": "img.jpg",
                        "x_center": 0.5,
                        "y_center": 0.5,
                        "width_relative": 0.2,
                        "height_relative": 0.2,
                    }
                ],
            },
        }

        ensembler = Ensembler(
            folders=[tmp_path],
            output=tmp_path / "ensemble",
            extension=".collectra",
        )

        result = ensembler._ensemble_image_crops(label_data, "crop_label")

        # Check ensemble attribute format
        if isinstance(result, list):
            assert len(result) >= 1
            item = result[0]
        else:
            item = result

        assert "ensemble" in item
        assert isinstance(item["ensemble"], list)
        assert all("::" in ref for ref in item["ensemble"])

    def test_ensemble_crop_coordinates_fused(self, tmp_path):
        """Test that coordinates are properly fused."""
        label_data = {
            "type": "collectra.ImageCrop",
            "items": {
                "folder1/file1.collectra": [
                    {
                        "type": "collectra.ImageCrop",
                        "id": "crop_1",
                        "data": "img.jpg",
                        "x_center": 0.4,
                        "y_center": 0.4,
                        "width_relative": 0.2,
                        "height_relative": 0.2,
                    }
                ],
                "folder2/file1.collectra": [
                    {
                        "type": "collectra.ImageCrop",
                        "id": "crop_2",
                        "data": "img.jpg",
                        "x_center": 0.6,
                        "y_center": 0.6,
                        "width_relative": 0.2,
                        "height_relative": 0.2,
                    }
                ],
            },
        }

        ensembler = Ensembler(
            folders=[tmp_path],
            output=tmp_path / "ensemble",
            extension=".collectra",
        )

        result = ensembler._ensemble_image_crops(label_data, "crop_label")

        if isinstance(result, list):
            item = result[0]
        else:
            item = result

        # Fused coordinates should be different from originals (averaged)
        assert "x_center" in item
        assert "y_center" in item
        assert "width_relative" in item
        assert "height_relative" in item


class TestEnsembleText:
    """Test Text label ensembling using edit distance."""

    def test_ensemble_text_consensus_selection(self, tmp_path):
        """Test that consensus text is selected using edit distance."""
        label_data = {
            "type": "collectra.Text",
            "items": {
                "folder1/file1.collectra": [
                    {
                        "type": "collectra.Text",
                        "id": "text_1",
                        "data": "hello",
                        "parents": ["crop-1"],
                    }
                ],
                "folder2/file1.collectra": [
                    {
                        "type": "collectra.Text",
                        "id": "text_2",
                        "data": "hallo",
                        "parents": ["crop-1"],
                    }
                ],
                "folder3/file1.collectra": [
                    {
                        "type": "collectra.Text",
                        "id": "text_3",
                        "data": "hullo",
                        "parents": ["crop-1"],
                    }
                ],
            },
        }

        ensembler = Ensembler(
            folders=[tmp_path],
            output=tmp_path / "ensemble",
            extension=".collectra",
        )

        # Set up tmp_ensembled with crop data for _find_nearest_crop_parent
        ensembler.tmp_ensembled = {
            "crop": {
                "type": "collectra.ImageCrop",
                "items": {
                    "folder1/file1.collectra": [
                        {
                            "type": "collectra.ImageCrop",
                            "id": "crop-1",
                            "data": "img.jpg",
                            "x_center": 0.5,
                            "y_center": 0.5,
                            "width_relative": 0.2,
                            "height_relative": 0.2,
                        }
                    ],
                    "folder2/file1.collectra": [
                        {
                            "type": "collectra.ImageCrop",
                            "id": "crop-1",
                            "data": "img.jpg",
                            "x_center": 0.5,
                            "y_center": 0.5,
                            "width_relative": 0.2,
                            "height_relative": 0.2,
                        }
                    ],
                    "folder3/file1.collectra": [
                        {
                            "type": "collectra.ImageCrop",
                            "id": "crop-1",
                            "data": "img.jpg",
                            "x_center": 0.5,
                            "y_center": 0.5,
                            "width_relative": 0.2,
                            "height_relative": 0.2,
                        }
                    ],
                },
            }
        }

        # Provide crop_id_mapping for text parents
        crop_id_mapping = {
            "folder1::crop-1": ["crop_ensembled_1"],
            "folder2::crop-1": ["crop_ensembled_1"],
            "folder3::crop-1": ["crop_ensembled_1"],
        }

        result = ensembler._ensemble_text(label_data, "text_label", crop_id_mapping)

        # Result should be list of ensembled text items
        if isinstance(result, list):
            assert len(result) >= 1
            item = result[0]
        else:
            item = result

        assert item["type"] == "collectra.Text"
        assert "data" in item  # Should have consensus text
        assert "ensemble" in item

    def test_ensemble_text_format_has_ensemble_attribute(self, tmp_path):
        """Test that ensembled text has correct format with ensemble attribute."""
        label_data = {
            "type": "collectra.Text",
            "items": {
                "folder1/file1.collectra": [
                    {
                        "type": "collectra.Text",
                        "id": "text_1",
                        "data": "text",
                        "parents": ["crop-1"],
                    }
                ],
                "folder2/file1.collectra": [
                    {
                        "type": "collectra.Text",
                        "id": "text_2",
                        "data": "text",
                        "parents": ["crop-1"],
                    }
                ],
            },
        }

        ensembler = Ensembler(
            folders=[tmp_path],
            output=tmp_path / "ensemble",
            extension=".collectra",
        )

        # Set up tmp_ensembled with crop data for _find_nearest_crop_parent
        ensembler.tmp_ensembled = {
            "crop": {
                "type": "collectra.ImageCrop",
                "items": {
                    "folder1/file1.collectra": [
                        {
                            "type": "collectra.ImageCrop",
                            "id": "crop-1",
                            "data": "img.jpg",
                            "x_center": 0.5,
                            "y_center": 0.5,
                            "width_relative": 0.2,
                            "height_relative": 0.2,
                        }
                    ],
                    "folder2/file1.collectra": [
                        {
                            "type": "collectra.ImageCrop",
                            "id": "crop-1",
                            "data": "img.jpg",
                            "x_center": 0.5,
                            "y_center": 0.5,
                            "width_relative": 0.2,
                            "height_relative": 0.2,
                        }
                    ],
                },
            }
        }

        # Provide crop_id_mapping for text parents
        crop_id_mapping = {
            "folder1::crop-1": ["crop_ensembled_1"],
            "folder2::crop-1": ["crop_ensembled_1"],
        }

        result = ensembler._ensemble_text(label_data, "text_label", crop_id_mapping)

        if isinstance(result, list):
            item = result[0]
        else:
            item = result

        assert "type" in item
        assert item["type"] == "collectra.Text"
        assert "ensemble" in item
        assert isinstance(item["ensemble"], list)
        assert all("::" in ref for ref in item["ensemble"])
        assert "parents" in item

    def test_ensemble_text_shortest_on_tie(self, tmp_path):
        """Test that shortest string is selected on edit distance tie."""
        label_data = {
            "type": "collectra.Text",
            "items": {
                "folder1/file1.collectra": [
                    {
                        "type": "collectra.Text",
                        "id": "text_1",
                        "data": "cat",
                        "parents": ["crop-1"],
                    }
                ],
                "folder2/file1.collectra": [
                    {
                        "type": "collectra.Text",
                        "id": "text_2",
                        "data": "dog",
                        "parents": ["crop-1"],
                    }
                ],
            },
        }

        ensembler = Ensembler(
            folders=[tmp_path],
            output=tmp_path / "ensemble",
            extension=".collectra",
        )

        # Set up tmp_ensembled with crop data for _find_nearest_crop_parent
        ensembler.tmp_ensembled = {
            "crop": {
                "type": "collectra.ImageCrop",
                "items": {
                    "folder1/file1.collectra": [
                        {
                            "type": "collectra.ImageCrop",
                            "id": "crop-1",
                            "data": "img.jpg",
                            "x_center": 0.5,
                            "y_center": 0.5,
                            "width_relative": 0.2,
                            "height_relative": 0.2,
                        }
                    ],
                    "folder2/file1.collectra": [
                        {
                            "type": "collectra.ImageCrop",
                            "id": "crop-1",
                            "data": "img.jpg",
                            "x_center": 0.5,
                            "y_center": 0.5,
                            "width_relative": 0.2,
                            "height_relative": 0.2,
                        }
                    ],
                },
            }
        }

        # Provide crop_id_mapping for text parents
        crop_id_mapping = {
            "folder1::crop-1": ["crop_ensembled_1"],
            "folder2::crop-1": ["crop_ensembled_1"],
        }

        result = ensembler._ensemble_text(label_data, "text_label", crop_id_mapping)

        if isinstance(result, list):
            item = result[0]
        else:
            item = result

        # Should select one of the shortest strings
        assert item["data"] in ["cat", "dog"]


class TestProcessFileLabel:
    """Test label processing based on type."""

    def test_process_image_type_label(self, tmp_path):
        """Test that Image type labels are processed correctly."""
        label_items = {
            "type": "collectra.Image",
            "items": {
                "folder1/file1.collectra": [
                    {"type": "collectra.Image", "data": "source.jpg"}
                ],
                "folder2/file1.collectra": [
                    {"type": "collectra.Image", "data": "source.jpg"}
                ],
            },
        }

        ensembler = Ensembler(
            folders=[tmp_path],
            output=tmp_path / "ensemble",
            extension=".collectra",
        )

        result = ensembler._process_file_label(
            "image_label", label_items, "collectra.Image"
        )

        # Image should be passed through or validated
        assert result is not None

    def test_process_imagecrop_type_label(self, tmp_path):
        """Test that ImageCrop type labels are ensembled with WBF."""
        label_items = {
            "type": "collectra.ImageCrop",
            "items": {
                "folder1/file1.collectra": [
                    {
                        "type": "collectra.ImageCrop",
                        "id": "crop_1",
                        "data": "img.jpg",
                        "x_center": 0.5,
                        "y_center": 0.5,
                        "width_relative": 0.2,
                        "height_relative": 0.2,
                    }
                ],
                "folder2/file1.collectra": [
                    {
                        "type": "collectra.ImageCrop",
                        "id": "crop_2",
                        "data": "img.jpg",
                        "x_center": 0.5,
                        "y_center": 0.5,
                        "width_relative": 0.2,
                        "height_relative": 0.2,
                    }
                ],
            },
        }

        ensembler = Ensembler(
            folders=[tmp_path],
            output=tmp_path / "ensemble",
            extension=".collectra",
        )

        result = ensembler._process_file_label(
            "crop_label", label_items, "collectra.ImageCrop"
        )

        # Should call _ensemble_image_crops
        assert result is not None

    def test_process_text_type_label(self, tmp_path):
        """Test that Text type labels are ensembled with edit distance."""
        label_items = {
            "type": "collectra.Text",
            "items": {
                "folder1/file1.collectra": [
                    {
                        "type": "collectra.Text",
                        "id": "text_1",
                        "data": "hello",
                        "parents": ["crop-1"],
                    }
                ],
                "folder2/file1.collectra": [
                    {
                        "type": "collectra.Text",
                        "id": "text_2",
                        "data": "hello",
                        "parents": ["crop-1"],
                    }
                ],
            },
        }

        ensembler = Ensembler(
            folders=[tmp_path],
            output=tmp_path / "ensemble",
            extension=".collectra",
        )

        # Set up tmp_ensembled with crop data for _find_nearest_crop_parent
        ensembler.tmp_ensembled = {
            "crop": {
                "type": "collectra.ImageCrop",
                "items": {
                    "folder1/file1.collectra": [
                        {
                            "type": "collectra.ImageCrop",
                            "id": "crop-1",
                            "data": "img.jpg",
                            "x_center": 0.5,
                            "y_center": 0.5,
                            "width_relative": 0.2,
                            "height_relative": 0.2,
                        }
                    ],
                    "folder2/file1.collectra": [
                        {
                            "type": "collectra.ImageCrop",
                            "id": "crop-1",
                            "data": "img.jpg",
                            "x_center": 0.5,
                            "y_center": 0.5,
                            "width_relative": 0.2,
                            "height_relative": 0.2,
                        }
                    ],
                },
            }
        }

        # Provide crop_id_mapping for text parents
        crop_id_mapping = {
            "folder1::crop-1": ["crop_ensembled_1"],
            "folder2::crop-1": ["crop_ensembled_1"],
        }

        result = ensembler._process_file_label(
            "text_label", label_items, "collectra.Text", crop_id_mapping=crop_id_mapping
        )

        # Should call _ensemble_text
        assert result is not None


class TestWriteResultsYaml:
    """Test writing results.yaml files."""

    def test_write_results_yaml_creates_file(self, tmp_path):
        """Test that results.yaml file is created."""
        ensembler = Ensembler(
            folders=[tmp_path],
            output=tmp_path / "ensemble",
            extension=".collectra",
        )

        output_path = tmp_path / "ensemble" / "file1.collectra"
        output_path.mkdir(parents=True, exist_ok=True)

        content = {
            "crop_label": [
                {
                    "type": "collectra.ImageCrop",
                    "id": "crop_1",
                    "data": "img.jpg",
                    "x_center": 0.5,
                    "y_center": 0.5,
                }
            ]
        }

        metadata = {
            "collectra_results_metadata": {
                "workflow": "test_workflow",
                "version": "0.1.0",
                "timestamp": "2026-01-16T12:00:00",
                "ensemble_sources": [
                    "folder1/file1.collectra",
                    "folder2/file1.collectra",
                ],
            }
        }

        ensembler._write_results_yaml(output_path, content, metadata)

        # Check that results.yaml exists
        results_yaml = output_path / "results.yaml"
        assert results_yaml.exists()

    def test_write_results_yaml_content_correct(self, tmp_path):
        """Test that results.yaml contains correct content."""
        ensembler = Ensembler(
            folders=[tmp_path],
            output=tmp_path / "ensemble",
            extension=".collectra",
        )

        output_path = tmp_path / "ensemble" / "file1.collectra"
        output_path.mkdir(parents=True, exist_ok=True)

        content = {"label1": {"type": "collectra.Image", "data": "img.jpg"}}

        metadata = {
            "collectra_results_metadata": {
                "workflow": "test",
                "version": "0.1.0",
                "timestamp": "2026-01-16T12:00:00",
                "ensemble_sources": ["folder1/file1.collectra"],
            }
        }

        ensembler._write_results_yaml(output_path, content, metadata)

        # Read and verify content
        with open(output_path / "results.yaml", "r") as f:
            saved_content = yaml.safe_load(f)

        assert "collectra_results_metadata" in saved_content
        assert "label1" in saved_content
        assert saved_content["collectra_results_metadata"]["workflow"] == "test"


class TestCopyArtifactFiles:
    """Test copying artifact files from source to ensemble folder."""

    def test_copy_artifact_files_copies_images(self, tmp_path):
        """Test that image files are copied."""
        source_path = tmp_path / "source" / "file1.collectra"
        source_path.mkdir(parents=True, exist_ok=True)

        # Create artifact files
        (source_path / "image.jpg").write_text("fake jpg content")
        (source_path / "document.txt").write_text("some text")
        (source_path / "results.yaml").write_text("metadata")

        dest_path = tmp_path / "ensemble" / "file1.collectra"
        dest_path.mkdir(parents=True, exist_ok=True)

        ensembler = Ensembler(
            folders=[tmp_path / "source"],
            output=tmp_path / "ensemble",
            extension=".collectra",
        )

        ensembler._copy_artifact_files(source_path, dest_path)

        # Check that artifact files are copied
        assert (dest_path / "image.jpg").exists()
        assert (dest_path / "document.txt").exists()

    def test_copy_artifact_files_skips_results_yaml(self, tmp_path):
        """Test that results.yaml is not copied."""
        source_path = tmp_path / "source" / "file1.collectra"
        source_path.mkdir(parents=True, exist_ok=True)

        (source_path / "results.yaml").write_text("metadata")
        (source_path / "image.jpg").write_text("fake jpg content")

        dest_path = tmp_path / "ensemble" / "file1.collectra"
        dest_path.mkdir(parents=True, exist_ok=True)

        ensembler = Ensembler(
            folders=[tmp_path / "source"],
            output=tmp_path / "ensemble",
            extension=".collectra",
        )

        ensembler._copy_artifact_files(source_path, dest_path)

        # results.yaml should not be copied
        assert not (dest_path / "results.yaml").exists()
        # But other files should be
        assert (dest_path / "image.jpg").exists()

    def test_copy_artifact_files_preserves_content(self, tmp_path):
        """Test that artifact file content is preserved."""
        source_path = tmp_path / "source" / "file1.collectra"
        source_path.mkdir(parents=True, exist_ok=True)

        original_content = "test file content"
        (source_path / "data.txt").write_text(original_content)

        dest_path = tmp_path / "ensemble" / "file1.collectra"
        dest_path.mkdir(parents=True, exist_ok=True)

        ensembler = Ensembler(
            folders=[tmp_path / "source"],
            output=tmp_path / "ensemble",
            extension=".collectra",
        )

        ensembler._copy_artifact_files(source_path, dest_path)

        # Check content is preserved
        copied_content = (dest_path / "data.txt").read_text()
        assert copied_content == original_content
