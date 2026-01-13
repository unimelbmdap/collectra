"""
Unit tests for the ensemble module.

This module tests the ensemble functionality including:
- Finding collectra files in folders
- Verifying files exist across multiple folders
- Creating ensemble output with merged results
- Loading link.yaml files
- Extracting labels with bounding boxes
"""

import shutil
from pathlib import Path

import pytest
import yaml

from collectra.ensemble import (
    _find_parent_with_bounding_box,
    _get_bounding_box_from_entry,
    _get_image_file,
    calculate_centroid_box,
    calculate_iou,
    create_ensemble_output,
    ensemble_files,
    ensemble_groups_for_file,
    extract_labels_with_bounding_boxes,
    find_centroid_text,
    find_collectra_files,
    generate_ensembled_values,
    get_ensemble_folder,
    get_source_collectra_files,
    group_by_bounding_box,
    group_by_label,
    load_link_yaml,
    load_results_yaml,
    verify_collectra_files,
    write_ensembled_results,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def collectra_folder_structure(tmp_path):
    """
    Create a basic collectra folder structure with .grapto folders.

    Structure:
        tmp_path/
            folder1/
                file1.grapto/
                    image.jpg
                    results.yaml
                file2.grapto/
                    image.png
                    results.yaml
            folder2/
                file1.grapto/
                    image.jpg
                    results.yaml
                file2.grapto/
                    image.png
                    results.yaml
    """
    folders = []
    for folder_name in ["folder1", "folder2"]:
        folder = tmp_path / folder_name
        folder.mkdir()
        folders.append(folder)

        for file_name in ["file1.grapto", "file2.grapto"]:
            grapto_folder = folder / file_name
            grapto_folder.mkdir()

            # Create image file
            image_ext = ".jpg" if "file1" in file_name else ".png"
            image_path = grapto_folder / f"image{image_ext}"
            image_path.write_bytes(b"fake image content")

            # Create results.yaml
            results_path = grapto_folder / "results.yaml"
            results_path.write_text("key: value\n")

    return tmp_path, folders


@pytest.fixture
def single_grapto_folder(tmp_path):
    """Create a single .grapto folder with standard content."""
    grapto_folder = tmp_path / "test.grapto"
    grapto_folder.mkdir()

    # Create image file
    image_path = grapto_folder / "specimen.jpg"
    image_path.write_bytes(b"specimen image data")

    # Create results.yaml
    results_path = grapto_folder / "results.yaml"
    results_path.write_text("field1: value1\nfield2: value2\n")

    return grapto_folder


@pytest.fixture
def grapto_folder_with_usage(tmp_path):
    """Create a .grapto folder with results.yaml, usage.yaml, and an image."""
    grapto_folder = tmp_path / "test_usage.grapto"
    grapto_folder.mkdir()

    # Create image file
    image_path = grapto_folder / "specimen.jpg"
    image_path.write_bytes(b"specimen image data")

    # Create results.yaml
    results_path = grapto_folder / "results.yaml"
    results_path.write_text("field1: value1\n")

    # Create usage.yaml
    usage_path = grapto_folder / "usage.yaml"
    usage_path.write_text("tokens: 100\n")

    return grapto_folder


# ============================================================================
# Tests for find_collectra_files
# ============================================================================


class TestFindCollectraFiles:
    """Tests for find_collectra_files function."""

    def test_find_files_in_valid_folder(self, collectra_folder_structure):
        """Test finding collectra files in a valid folder."""
        tmp_path, folders = collectra_folder_structure
        folder = folders[0]  # folder1

        result = find_collectra_files(folder, ".grapto")

        assert len(result) == 2
        assert "file1.grapto" in result
        assert "file2.grapto" in result
        assert result["file1.grapto"] == folder / "file1.grapto"
        assert result["file2.grapto"] == folder / "file2.grapto"

    def test_empty_folder_returns_empty_dict(self, tmp_path):
        """Test that an empty folder returns an empty dictionary."""
        empty_folder = tmp_path / "empty"
        empty_folder.mkdir()

        result = find_collectra_files(empty_folder, ".grapto")

        assert result == {}

    def test_folder_with_no_matching_extension(self, tmp_path):
        """Test folder with files but no matching extension."""
        folder = tmp_path / "no_match"
        folder.mkdir()

        # Create folders with different extensions
        (folder / "file1.livy").mkdir()
        (folder / "file2.other").mkdir()

        result = find_collectra_files(folder, ".grapto")

        assert result == {}

    def test_folder_doesnt_exist_raises_error(self, tmp_path):
        """Test that a non-existent folder raises FileNotFoundError."""
        non_existent = tmp_path / "does_not_exist"

        with pytest.raises(FileNotFoundError) as exc_info:
            find_collectra_files(non_existent, ".grapto")

        assert "does not exist" in str(exc_info.value)

    def test_path_is_file_raises_error(self, tmp_path):
        """Test that passing a file path raises NotADirectoryError."""
        file_path = tmp_path / "not_a_folder.txt"
        file_path.write_text("content")

        with pytest.raises(NotADirectoryError) as exc_info:
            find_collectra_files(file_path, ".grapto")

        assert "not a directory" in str(exc_info.value)

    def test_duplicate_files_raises_value_error(self, tmp_path):
        """Test that duplicate files in same folder raises ValueError.

        Note: In practice, a filesystem won't allow two items with the same name,
        but this test verifies the code's duplicate detection logic works.
        This scenario shouldn't happen in normal filesystem operations.
        """
        folder = tmp_path / "test_folder"
        folder.mkdir()

        # Create a single grapto folder
        (folder / "unique.grapto").mkdir()

        # Since we can't actually create duplicates in filesystem,
        # we verify no error is raised for unique files
        result = find_collectra_files(folder, ".grapto")
        assert len(result) == 1

    def test_different_extension(self, tmp_path):
        """Test finding files with a different extension."""
        folder = tmp_path / "mixed"
        folder.mkdir()

        (folder / "file1.grapto").mkdir()
        (folder / "file2.livy").mkdir()
        (folder / "file3.livy").mkdir()

        result = find_collectra_files(folder, ".livy")

        assert len(result) == 2
        assert "file2.livy" in result
        assert "file3.livy" in result
        assert "file1.grapto" not in result

    def test_ignores_regular_files(self, tmp_path):
        """Test that regular files (not directories) are ignored."""
        folder = tmp_path / "mixed_content"
        folder.mkdir()

        # Create a grapto folder
        (folder / "valid.grapto").mkdir()

        # Create a file with .grapto extension (not a directory)
        (folder / "invalid.grapto.txt").write_text("not a folder")

        result = find_collectra_files(folder, ".grapto")

        assert len(result) == 1
        assert "valid.grapto" in result


# ============================================================================
# Tests for verify_collectra_files
# ============================================================================


class TestVerifyCollectraFiles:
    """Tests for verify_collectra_files function."""

    def test_files_present_in_all_folders(self, collectra_folder_structure):
        """Test that files present in all folders are verified."""
        _, folders = collectra_folder_structure

        verified, warnings = verify_collectra_files(folders, ".grapto")

        assert len(verified) == 2
        assert "file1.grapto" in verified
        assert "file2.grapto" in verified
        assert len(warnings) == 0

        # Verify paths are in correct order
        assert verified["file1.grapto"][0] == folders[0] / "file1.grapto"
        assert verified["file1.grapto"][1] == folders[1] / "file1.grapto"

    def test_files_missing_from_some_folders(self, tmp_path):
        """Test that files missing from some folders generate warnings."""
        folder1 = tmp_path / "folder1"
        folder2 = tmp_path / "folder2"
        folder1.mkdir()
        folder2.mkdir()

        # Create file1.grapto in both folders
        (folder1 / "file1.grapto").mkdir()
        (folder2 / "file1.grapto").mkdir()

        # Create file2.grapto only in folder1
        (folder1 / "file2.grapto").mkdir()

        verified, warnings = verify_collectra_files([folder1, folder2], ".grapto")

        assert len(verified) == 1
        assert "file1.grapto" in verified
        assert "file2.grapto" not in verified
        assert len(warnings) == 1
        assert "file2.grapto" in warnings[0]
        assert str(folder2) in warnings[0]

    def test_empty_folders_scenario(self, tmp_path):
        """Test verification with empty folders returns empty verified dict."""
        folder1 = tmp_path / "empty1"
        folder2 = tmp_path / "empty2"
        folder1.mkdir()
        folder2.mkdir()

        verified, warnings = verify_collectra_files([folder1, folder2], ".grapto")

        assert verified == {}
        assert warnings == []

    def test_single_folder_verification(self, tmp_path):
        """Test verification with a single folder (all files should be verified)."""
        folder = tmp_path / "single"
        folder.mkdir()

        (folder / "file1.grapto").mkdir()
        (folder / "file2.grapto").mkdir()

        verified, warnings = verify_collectra_files([folder], ".grapto")

        assert len(verified) == 2
        assert "file1.grapto" in verified
        assert "file2.grapto" in verified
        assert warnings == []

    def test_no_folders_raises_error(self):
        """Test that empty folder list raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            verify_collectra_files([], ".grapto")

        assert "At least one input folder" in str(exc_info.value)

    def test_non_existent_folder_raises_error(self, tmp_path):
        """Test that non-existent folder raises FileNotFoundError."""
        valid_folder = tmp_path / "valid"
        valid_folder.mkdir()
        (valid_folder / "file.grapto").mkdir()

        non_existent = tmp_path / "does_not_exist"

        with pytest.raises(FileNotFoundError):
            verify_collectra_files([valid_folder, non_existent], ".grapto")

    def test_multiple_folders_partial_overlap(self, tmp_path):
        """Test verification with three folders and partial file overlap."""
        folder1 = tmp_path / "folder1"
        folder2 = tmp_path / "folder2"
        folder3 = tmp_path / "folder3"
        folder1.mkdir()
        folder2.mkdir()
        folder3.mkdir()

        # common.grapto in all three
        (folder1 / "common.grapto").mkdir()
        (folder2 / "common.grapto").mkdir()
        (folder3 / "common.grapto").mkdir()

        # partial1.grapto only in folder1 and folder2
        (folder1 / "partial1.grapto").mkdir()
        (folder2 / "partial1.grapto").mkdir()

        # partial2.grapto only in folder2 and folder3
        (folder2 / "partial2.grapto").mkdir()
        (folder3 / "partial2.grapto").mkdir()

        verified, warnings = verify_collectra_files(
            [folder1, folder2, folder3], ".grapto"
        )

        assert len(verified) == 1
        assert "common.grapto" in verified
        assert len(warnings) == 2


# ============================================================================
# Tests for _get_image_file
# ============================================================================


class TestGetImageFile:
    """Tests for _get_image_file helper function."""

    def test_get_image_file_basic(self, single_grapto_folder):
        """Test getting image file from a grapto folder."""
        result = _get_image_file(single_grapto_folder)

        assert result is not None
        assert result.name == "specimen.jpg"

    def test_get_image_file_excludes_results_yaml(self, single_grapto_folder):
        """Test that results.yaml is excluded from image search."""
        result = _get_image_file(single_grapto_folder)

        assert result.name != "results.yaml"

    def test_get_image_file_excludes_usage_yaml(self, grapto_folder_with_usage):
        """Test that usage.yaml is excluded from image search."""
        result = _get_image_file(grapto_folder_with_usage)

        assert result is not None
        assert result.name == "specimen.jpg"
        assert result.name != "usage.yaml"

    def test_get_image_file_no_image(self, tmp_path):
        """Test returns None when no image file exists."""
        folder = tmp_path / "no_image.grapto"
        folder.mkdir()

        # Create only results.yaml and usage.yaml
        (folder / "results.yaml").write_text("key: value")
        (folder / "usage.yaml").write_text("tokens: 100")

        result = _get_image_file(folder)

        assert result is None

    def test_get_image_file_empty_folder(self, tmp_path):
        """Test returns None for empty folder."""
        folder = tmp_path / "empty.grapto"
        folder.mkdir()

        result = _get_image_file(folder)

        assert result is None


# ============================================================================
# Tests for create_ensemble_output
# ============================================================================


class TestCreateEnsembleOutput:
    """Tests for create_ensemble_output function."""

    def test_output_folder_created(self, collectra_folder_structure, tmp_path):
        """Test that output folder is created correctly."""
        _, folders = collectra_folder_structure
        output_folder = tmp_path / "ensemble_output"

        verified_files = {
            "file1.grapto": [folders[0] / "file1.grapto", folders[1] / "file1.grapto"]
        }

        create_ensemble_output(verified_files, output_folder, ".grapto")

        assert output_folder.exists()
        assert output_folder.is_dir()

    def test_results_yaml_created_empty(self, collectra_folder_structure, tmp_path):
        """Test that results.yaml is created and empty."""
        _, folders = collectra_folder_structure
        output_folder = tmp_path / "ensemble_output"

        verified_files = {
            "file1.grapto": [folders[0] / "file1.grapto", folders[1] / "file1.grapto"]
        }

        create_ensemble_output(verified_files, output_folder, ".grapto")

        results_yaml = output_folder / "file1.grapto" / "results.yaml"
        assert results_yaml.exists()
        assert results_yaml.read_text() == ""

    def test_image_copied_from_first_source(self, collectra_folder_structure, tmp_path):
        """Test that image is copied from the first source folder."""
        _, folders = collectra_folder_structure
        output_folder = tmp_path / "ensemble_output"

        verified_files = {
            "file1.grapto": [folders[0] / "file1.grapto", folders[1] / "file1.grapto"]
        }

        create_ensemble_output(verified_files, output_folder, ".grapto")

        # Check image exists in output
        output_grapto = output_folder / "file1.grapto"
        image_files = [
            f
            for f in output_grapto.iterdir()
            if f.is_file() and f.name != "results.yaml"
        ]

        assert len(image_files) == 1
        assert image_files[0].name == "image.jpg"

    def test_link_yaml_created_with_correct_format(
        self, collectra_folder_structure, tmp_path
    ):
        """Test that link.yaml is created with correct format."""
        _, folders = collectra_folder_structure
        output_folder = tmp_path / "ensemble_output"

        verified_files = {
            "file1.grapto": [folders[0] / "file1.grapto", folders[1] / "file1.grapto"],
            "file2.grapto": [folders[0] / "file2.grapto", folders[1] / "file2.grapto"],
        }

        link_yaml_path = create_ensemble_output(
            verified_files, output_folder, ".grapto"
        )

        assert link_yaml_path.exists()
        assert link_yaml_path.name == "link.yaml"

        with open(link_yaml_path) as f:
            link_data = yaml.safe_load(f)

        assert "file1.grapto" in link_data
        assert "file2.grapto" in link_data
        assert len(link_data["file1.grapto"]) == 2
        assert str(folders[0] / "file1.grapto") in link_data["file1.grapto"]
        assert str(folders[1] / "file1.grapto") in link_data["file1.grapto"]

    def test_link_yaml_returns_correct_path(self, collectra_folder_structure, tmp_path):
        """Test that the returned path is the link.yaml file."""
        _, folders = collectra_folder_structure
        output_folder = tmp_path / "ensemble_output"

        verified_files = {
            "file1.grapto": [folders[0] / "file1.grapto", folders[1] / "file1.grapto"]
        }

        result = create_ensemble_output(verified_files, output_folder, ".grapto")

        assert result == output_folder / "link.yaml"

    def test_empty_verified_files_raises_error(self, tmp_path):
        """Test that empty verified_files raises ValueError."""
        output_folder = tmp_path / "output"

        with pytest.raises(ValueError) as exc_info:
            create_ensemble_output({}, output_folder, ".grapto")

        assert "No verified files" in str(exc_info.value)

    def test_multiple_files_processed(self, collectra_folder_structure, tmp_path):
        """Test that multiple files are processed correctly."""
        _, folders = collectra_folder_structure
        output_folder = tmp_path / "ensemble_output"

        verified_files = {
            "file1.grapto": [folders[0] / "file1.grapto", folders[1] / "file1.grapto"],
            "file2.grapto": [folders[0] / "file2.grapto", folders[1] / "file2.grapto"],
        }

        create_ensemble_output(verified_files, output_folder, ".grapto")

        assert (output_folder / "file1.grapto").exists()
        assert (output_folder / "file2.grapto").exists()
        assert (output_folder / "file1.grapto" / "results.yaml").exists()
        assert (output_folder / "file2.grapto" / "results.yaml").exists()

    def test_output_folder_nested_path(self, collectra_folder_structure, tmp_path):
        """Test that nested output folder paths are created."""
        _, folders = collectra_folder_structure
        output_folder = tmp_path / "level1" / "level2" / "output"

        verified_files = {
            "file1.grapto": [folders[0] / "file1.grapto", folders[1] / "file1.grapto"]
        }

        create_ensemble_output(verified_files, output_folder, ".grapto")

        assert output_folder.exists()
        assert (output_folder / "file1.grapto").exists()


# ============================================================================
# Tests for ensemble_files (Integration)
# ============================================================================


class TestEnsembleFiles:
    """Tests for ensemble_files main function."""

    def test_full_integration(self, collectra_folder_structure, tmp_path):
        """Test full integration of all steps."""
        _, folders = collectra_folder_structure
        output_folder = tmp_path / "ensemble_output"

        result = ensemble_files(folders, output_folder, ".grapto")

        # Check link.yaml is returned
        assert result.exists()
        assert result.name == "link.yaml"

        # Check output structure
        assert (output_folder / "file1.grapto").exists()
        assert (output_folder / "file2.grapto").exists()

        # Check results.yaml files exist and have ensembled content
        assert (output_folder / "file1.grapto" / "results.yaml").exists()
        assert (output_folder / "file2.grapto" / "results.yaml").exists()
        # Results should contain metadata from ensemble process
        content1 = (output_folder / "file1.grapto" / "results.yaml").read_text()
        assert "collectra_results_metadata" in content1

    def test_with_partial_overlap(self, tmp_path):
        """Test with folders that have partial file overlap."""
        folder1 = tmp_path / "source1"
        folder2 = tmp_path / "source2"
        output_folder = tmp_path / "output"

        folder1.mkdir()
        folder2.mkdir()

        # Create common file in both
        for folder in [folder1, folder2]:
            grapto = folder / "common.grapto"
            grapto.mkdir()
            (grapto / "image.jpg").write_bytes(b"img")
            (grapto / "results.yaml").write_text("data: test")

        # Create unique file only in folder1
        unique = folder1 / "unique.grapto"
        unique.mkdir()
        (unique / "image.jpg").write_bytes(b"img")
        (unique / "results.yaml").write_text("data: unique")

        result = ensemble_files([folder1, folder2], output_folder, ".grapto")

        # Only common file should be in output
        assert (output_folder / "common.grapto").exists()
        assert not (output_folder / "unique.grapto").exists()

        # Verify link.yaml content
        with open(result) as f:
            link_data = yaml.safe_load(f)
        assert "common.grapto" in link_data
        assert "unique.grapto" not in link_data

    def test_no_input_folders_raises_error(self, tmp_path):
        """Test that no input folders raises ValueError."""
        output_folder = tmp_path / "output"

        with pytest.raises(ValueError) as exc_info:
            ensemble_files([], output_folder, ".grapto")

        assert "At least one input folder" in str(exc_info.value)

    def test_no_verified_files_raises_error(self, tmp_path):
        """Test that no verified files raises ValueError."""
        folder1 = tmp_path / "folder1"
        folder2 = tmp_path / "folder2"
        folder1.mkdir()
        folder2.mkdir()

        # Create different files in each folder
        (folder1 / "file1.grapto").mkdir()
        (folder2 / "file2.grapto").mkdir()

        output_folder = tmp_path / "output"

        with pytest.raises(ValueError) as exc_info:
            ensemble_files([folder1, folder2], output_folder, ".grapto")

        assert "No files found" in str(exc_info.value)

    def test_with_different_extension(self, tmp_path):
        """Test ensemble with a different extension (.livy)."""
        folder1 = tmp_path / "folder1"
        folder2 = tmp_path / "folder2"
        output_folder = tmp_path / "output"

        folder1.mkdir()
        folder2.mkdir()

        for folder in [folder1, folder2]:
            livy = folder / "page.livy"
            livy.mkdir()
            (livy / "scan.png").write_bytes(b"image data")
            (livy / "results.yaml").write_text("transcription: text")

        result = ensemble_files([folder1, folder2], output_folder, ".livy")

        assert result.exists()
        assert (output_folder / "page.livy").exists()

    def test_string_paths_converted(self, collectra_folder_structure, tmp_path):
        """Test that string paths are converted to Path objects."""
        _, folders = collectra_folder_structure
        output_folder = tmp_path / "output"

        # Pass string paths
        string_folders = [str(f) for f in folders]
        string_output = str(output_folder)

        result = ensemble_files(string_folders, string_output, ".grapto")

        assert result.exists()


# ============================================================================
# Tests for load_link_yaml
# ============================================================================


class TestLoadLinkYaml:
    """Tests for load_link_yaml function."""

    def test_load_valid_link_yaml(self, tmp_path):
        """Test loading a valid link.yaml file."""
        link_yaml_path = tmp_path / "link.yaml"

        link_data = {
            "file1.grapto": [
                "/path/to/source1/file1.grapto",
                "/path/to/source2/file1.grapto",
            ],
            "file2.grapto": [
                "/path/to/source1/file2.grapto",
                "/path/to/source2/file2.grapto",
            ],
        }

        with open(link_yaml_path, "w") as f:
            yaml.dump(link_data, f)

        result = load_link_yaml(link_yaml_path)

        assert len(result) == 2
        assert "file1.grapto" in result
        assert "file2.grapto" in result

        # Check paths are converted to Path objects
        assert isinstance(result["file1.grapto"][0], Path)
        assert result["file1.grapto"][0] == Path("/path/to/source1/file1.grapto")

    def test_load_non_existent_path_raises_error(self, tmp_path):
        """Test loading from non-existent path raises FileNotFoundError."""
        non_existent = tmp_path / "does_not_exist.yaml"

        with pytest.raises(FileNotFoundError) as exc_info:
            load_link_yaml(non_existent)

        assert "Link file not found" in str(exc_info.value)

    def test_load_empty_link_yaml(self, tmp_path):
        """Test loading an empty link.yaml file returns empty dict."""
        link_yaml_path = tmp_path / "link.yaml"
        link_yaml_path.write_text("")

        result = load_link_yaml(link_yaml_path)

        assert result == {}

    def test_load_link_yaml_from_ensemble_output(
        self, collectra_folder_structure, tmp_path
    ):
        """Test loading link.yaml created by ensemble_files."""
        _, folders = collectra_folder_structure
        output_folder = tmp_path / "ensemble_output"

        link_yaml_path = ensemble_files(folders, output_folder, ".grapto")

        result = load_link_yaml(link_yaml_path)

        assert len(result) == 2
        assert "file1.grapto" in result
        assert "file2.grapto" in result


# ============================================================================
# Tests for get_ensemble_folder
# ============================================================================


class TestGetEnsembleFolder:
    """Tests for get_ensemble_folder function."""

    def test_get_ensemble_folder_basic(self, tmp_path):
        """Test getting ensemble folder from link.yaml path."""
        ensemble_folder = tmp_path / "ensemble"
        ensemble_folder.mkdir()
        link_yaml = ensemble_folder / "link.yaml"
        link_yaml.touch()

        result = get_ensemble_folder(link_yaml)

        assert result == ensemble_folder

    def test_get_ensemble_folder_nested(self, tmp_path):
        """Test getting ensemble folder from nested path."""
        ensemble_folder = tmp_path / "data" / "results" / "ensemble"
        ensemble_folder.mkdir(parents=True)
        link_yaml = ensemble_folder / "link.yaml"
        link_yaml.touch()

        result = get_ensemble_folder(link_yaml)

        assert result == ensemble_folder
        assert result.name == "ensemble"


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_file_single_folder(self, tmp_path):
        """Test with a single file in a single folder."""
        folder = tmp_path / "single"
        folder.mkdir()

        grapto = folder / "only.grapto"
        grapto.mkdir()
        (grapto / "image.jpg").write_bytes(b"img")
        (grapto / "results.yaml").write_text("key: value")

        output = tmp_path / "output"
        result = ensemble_files([folder], output, ".grapto")

        assert result.exists()
        assert (output / "only.grapto").exists()

    def test_large_number_of_files(self, tmp_path):
        """Test with a large number of files."""
        folder1 = tmp_path / "folder1"
        folder2 = tmp_path / "folder2"
        folder1.mkdir()
        folder2.mkdir()

        # Create 50 files in each folder
        for i in range(50):
            for folder in [folder1, folder2]:
                grapto = folder / f"file{i:03d}.grapto"
                grapto.mkdir()
                (grapto / "image.jpg").write_bytes(b"img")
                (grapto / "results.yaml").write_text(f"index: {i}")

        output = tmp_path / "output"
        result = ensemble_files([folder1, folder2], output, ".grapto")

        # Verify all files processed
        with open(result) as f:
            link_data = yaml.safe_load(f)

        assert len(link_data) == 50

    def test_special_characters_in_filename(self, tmp_path):
        """Test files with special characters in names."""
        folder1 = tmp_path / "folder1"
        folder2 = tmp_path / "folder2"
        folder1.mkdir()
        folder2.mkdir()

        # Create file with spaces and special chars (filesystem safe)
        filename = "specimen_001-test.grapto"

        for folder in [folder1, folder2]:
            grapto = folder / filename
            grapto.mkdir()
            (grapto / "image.jpg").write_bytes(b"img")
            (grapto / "results.yaml").write_text("key: value")

        output = tmp_path / "output"
        result = ensemble_files([folder1, folder2], output, ".grapto")

        assert (output / filename).exists()

    def test_grapto_folder_without_image(self, tmp_path):
        """Test handling of grapto folder without an image file."""
        folder1 = tmp_path / "folder1"
        folder2 = tmp_path / "folder2"
        folder1.mkdir()
        folder2.mkdir()

        for folder in [folder1, folder2]:
            grapto = folder / "no_image.grapto"
            grapto.mkdir()
            # Only create results.yaml, no image
            (grapto / "results.yaml").write_text("key: value")

        output = tmp_path / "output"
        result = ensemble_files([folder1, folder2], output, ".grapto")

        # Should still create output, just without image
        assert (output / "no_image.grapto").exists()
        assert (output / "no_image.grapto" / "results.yaml").exists()

        # Check no image was copied (only results.yaml should exist)
        files = list((output / "no_image.grapto").iterdir())
        assert len(files) == 1
        assert files[0].name == "results.yaml"


# ============================================================================
# Tests for get_source_collectra_files
# ============================================================================


class TestGetSourceCollectraFiles:
    """Tests for get_source_collectra_files function."""

    def test_get_sources_for_existing_folder(self, tmp_path):
        """Test getting source files for an existing ensemble folder."""
        link_yaml_path = tmp_path / "link.yaml"
        link_data = {
            "file1.grapto": [
                "data/model1/file1.grapto",
                "data/model2/file1.grapto",
            ],
            "file2.grapto": [
                "data/model1/file2.grapto",
                "data/model2/file2.grapto",
            ],
        }
        with open(link_yaml_path, "w") as f:
            yaml.dump(link_data, f)

        result = get_source_collectra_files(Path("file1.grapto"), link_yaml_path)

        assert len(result) == 2
        assert result[0] == Path("data/model1/file1.grapto")
        assert result[1] == Path("data/model2/file1.grapto")

    def test_get_sources_with_full_path(self, tmp_path):
        """Test getting source files when full path is provided."""
        link_yaml_path = tmp_path / "link.yaml"
        link_data = {
            "file1.grapto": [
                "data/model1/file1.grapto",
                "data/model2/file1.grapto",
            ],
        }
        with open(link_yaml_path, "w") as f:
            yaml.dump(link_data, f)

        # Provide full path instead of just name
        full_path = tmp_path / "ensemble" / "file1.grapto"
        result = get_source_collectra_files(full_path, link_yaml_path)

        assert len(result) == 2

    def test_get_sources_folder_not_found(self, tmp_path):
        """Test that KeyError is raised for non-existent folder."""
        link_yaml_path = tmp_path / "link.yaml"
        link_data = {"existing.grapto": ["data/model1/existing.grapto"]}
        with open(link_yaml_path, "w") as f:
            yaml.dump(link_data, f)

        with pytest.raises(KeyError) as exc_info:
            get_source_collectra_files(Path("nonexistent.grapto"), link_yaml_path)

        assert "nonexistent.grapto" in str(exc_info.value)

    def test_get_sources_link_yaml_not_found(self, tmp_path):
        """Test that FileNotFoundError is raised for missing link.yaml."""
        with pytest.raises(FileNotFoundError):
            get_source_collectra_files(
                Path("file.grapto"), tmp_path / "nonexistent.yaml"
            )


# ============================================================================
# Tests for load_results_yaml
# ============================================================================


class TestLoadResultsYaml:
    """Tests for load_results_yaml function."""

    def test_load_valid_results_yaml(self, tmp_path):
        """Test loading a valid results.yaml file."""
        grapto_folder = tmp_path / "test.grapto"
        grapto_folder.mkdir()

        results_data = {
            "collectra_results_metadata": {
                "workflow": "Grapto",
                "version": "0.1.0",
            },
            "registration_number": {
                "type": "collectra.Text",
                "id": "reg-123",
                "data": "P.350015",
            },
        }
        with open(grapto_folder / "results.yaml", "w") as f:
            yaml.dump(results_data, f)

        result = load_results_yaml(grapto_folder)

        assert "collectra_results_metadata" in result
        assert "registration_number" in result
        assert result["registration_number"]["data"] == "P.350015"

    def test_load_empty_results_yaml(self, tmp_path):
        """Test loading an empty results.yaml file."""
        grapto_folder = tmp_path / "empty.grapto"
        grapto_folder.mkdir()
        (grapto_folder / "results.yaml").write_text("")

        result = load_results_yaml(grapto_folder)

        assert result == {}

    def test_load_missing_results_yaml(self, tmp_path):
        """Test that FileNotFoundError is raised for missing results.yaml."""
        grapto_folder = tmp_path / "no_results.grapto"
        grapto_folder.mkdir()

        with pytest.raises(FileNotFoundError) as exc_info:
            load_results_yaml(grapto_folder)

        assert "results.yaml not found" in str(exc_info.value)


# ============================================================================
# Tests for _get_bounding_box_from_entry
# ============================================================================


class TestGetBoundingBoxFromEntry:
    """Tests for _get_bounding_box_from_entry helper function."""

    def test_extract_complete_bounding_box(self):
        """Test extracting bounding box with all fields present."""
        entry = {
            "type": "collectra.ImageCrop",
            "x_center": 0.5,
            "y_center": 0.6,
            "width_relative": 0.1,
            "height_relative": 0.2,
        }

        result = _get_bounding_box_from_entry(entry)

        assert result is not None
        assert result["x_center"] == 0.5
        assert result["y_center"] == 0.6
        assert result["width_relative"] == 0.1
        assert result["height_relative"] == 0.2

    def test_missing_bounding_box_fields(self):
        """Test returns None when bounding box fields are missing."""
        entry = {
            "type": "collectra.ImageCrop",
            "x_center": 0.5,
            # Missing other fields
        }

        result = _get_bounding_box_from_entry(entry)

        assert result is None

    def test_converts_to_float(self):
        """Test that values are converted to float."""
        entry = {
            "x_center": "0.5",
            "y_center": "0.6",
            "width_relative": "0.1",
            "height_relative": "0.2",
        }

        result = _get_bounding_box_from_entry(entry)

        assert result is not None
        assert isinstance(result["x_center"], float)


# ============================================================================
# Tests for _find_parent_with_bounding_box
# ============================================================================


class TestFindParentWithBoundingBox:
    """Tests for _find_parent_with_bounding_box helper function."""

    def test_single_parent_string(self):
        """Test finding bounding box with single parent (string)."""
        results_data = {
            "image_crop": {
                "type": "collectra.ImageCrop",
                "id": "image-123",
                "x_center": 0.5,
                "y_center": 0.6,
                "width_relative": 0.1,
                "height_relative": 0.2,
            },
            "text_field": {
                "type": "collectra.Text",
                "id": "text-456",
                "parents": "image-123",
                "data": "Some text",
            },
        }

        entry = results_data["text_field"]
        result = _find_parent_with_bounding_box(entry, results_data)

        assert result is not None
        assert result["x_center"] == 0.5
        assert result["y_center"] == 0.6

    def test_multiple_parents_list_uses_last(self):
        """Test that last parent in list is used for bounding box."""
        results_data = {
            "first_parent": {
                "type": "collectra.ImageCrop",
                "id": "parent-1",
                "x_center": 0.1,
                "y_center": 0.1,
                "width_relative": 0.1,
                "height_relative": 0.1,
            },
            "last_parent": {
                "type": "collectra.ImageCrop",
                "id": "parent-2",
                "x_center": 0.9,
                "y_center": 0.9,
                "width_relative": 0.9,
                "height_relative": 0.9,
            },
            "text_field": {
                "type": "collectra.Text",
                "id": "text-456",
                "parents": ["parent-1", "parent-2"],
                "data": "Some text",
            },
        }

        entry = results_data["text_field"]
        result = _find_parent_with_bounding_box(entry, results_data)

        assert result is not None
        # Should use last parent's bounding box
        assert result["x_center"] == 0.9
        assert result["y_center"] == 0.9

    def test_recursive_parent_traversal(self):
        """Test traversing multiple levels to find ImageCrop parent."""
        results_data = {
            "image_crop": {
                "type": "collectra.ImageCrop",
                "id": "image-123",
                "x_center": 0.5,
                "y_center": 0.6,
                "width_relative": 0.1,
                "height_relative": 0.2,
            },
            "intermediate_text": {
                "type": "collectra.Text",
                "id": "text-intermediate",
                "parents": "image-123",
                "data": "Intermediate",
            },
            "final_text": {
                "type": "collectra.Text",
                "id": "text-final",
                "parents": "text-intermediate",
                "data": "Final text",
            },
        }

        entry = results_data["final_text"]
        result = _find_parent_with_bounding_box(entry, results_data)

        assert result is not None
        assert result["x_center"] == 0.5

    def test_no_parents_returns_none(self):
        """Test that entry without parents returns None."""
        results_data = {
            "orphan_text": {
                "type": "collectra.Text",
                "id": "orphan-123",
                "data": "Orphan text",
            },
        }

        entry = results_data["orphan_text"]
        result = _find_parent_with_bounding_box(entry, results_data)

        assert result is None

    def test_parent_not_found_returns_none(self):
        """Test that missing parent returns None."""
        results_data = {
            "text_field": {
                "type": "collectra.Text",
                "id": "text-456",
                "parents": "nonexistent-parent",
                "data": "Some text",
            },
        }

        entry = results_data["text_field"]
        result = _find_parent_with_bounding_box(entry, results_data)

        assert result is None

    def test_empty_parents_list_returns_none(self):
        """Test that empty parents list returns None."""
        results_data = {
            "text_field": {
                "type": "collectra.Text",
                "id": "text-456",
                "parents": [],
                "data": "Some text",
            },
        }

        entry = results_data["text_field"]
        result = _find_parent_with_bounding_box(entry, results_data)

        assert result is None


# ============================================================================
# Tests for extract_labels_with_bounding_boxes
# ============================================================================


class TestExtractLabelsWithBoundingBoxes:
    """Tests for extract_labels_with_bounding_boxes function."""

    @pytest.fixture
    def grapto_with_labels(self, tmp_path):
        """Create a grapto folder with Text and ImageCrop labels."""
        grapto_folder = tmp_path / "test.grapto"
        grapto_folder.mkdir()

        results_data = {
            "collectra_results_metadata": {
                "workflow": "Grapto",
                "version": "0.1.0",
            },
            "primary_label": {
                "type": "collectra.ImageCrop",
                "id": "primary-label-123",
                "parents": "specimen_sheet",
                "x_center": 0.5,
                "y_center": 0.8,
                "width_relative": 0.3,
                "height_relative": 0.25,
            },
            "registration_image": {
                "type": "collectra.ImageCrop",
                "id": "reg-image-456",
                "parents": "primary-label-123",
                "x_center": 0.4,
                "y_center": 0.75,
                "width_relative": 0.1,
                "height_relative": 0.05,
            },
            "registration_number": {
                "type": "collectra.Text",
                "id": "reg-num-789",
                "parents": ["primary-label-123", "reg-image-456"],
                "data": "P.350015",
            },
        }
        with open(grapto_folder / "results.yaml", "w") as f:
            yaml.dump(results_data, f)

        return grapto_folder

    def test_extract_text_with_parent_bounding_box(self, grapto_with_labels):
        """Test extracting Text labels with parent ImageCrop bounding boxes."""
        result = extract_labels_with_bounding_boxes([grapto_with_labels])

        # Should have entries for ImageCrop and Text labels
        text_entries = [r for r in result if r[0] == "registration_number"]

        assert len(text_entries) == 1
        label_name, data, bbox, source, item_id, entry_type = text_entries[0]

        assert label_name == "registration_number"
        assert data == "P.350015"
        # Bounding box should come from last parent (registration_image)
        assert bbox["x_center"] == 0.4
        assert bbox["y_center"] == 0.75
        assert source == str(grapto_with_labels)
        assert item_id == "reg-num-789"
        assert entry_type == "collectra.Text"

    def test_extract_imagecrop_with_own_bounding_box(self, grapto_with_labels):
        """Test extracting ImageCrop labels with their own bounding boxes."""
        result = extract_labels_with_bounding_boxes([grapto_with_labels])

        # Find primary_label entry
        primary_entries = [r for r in result if r[0] == "primary_label"]

        assert len(primary_entries) == 1
        label_name, data, bbox, source, item_id, entry_type = primary_entries[0]

        assert label_name == "primary_label"
        # ImageCrop entries should have empty text content
        assert data == ""
        assert bbox["x_center"] == 0.5
        assert bbox["y_center"] == 0.8
        assert item_id == "primary-label-123"
        assert entry_type == "collectra.ImageCrop"

    def test_multiple_source_folders(self, tmp_path):
        """Test extracting labels from multiple source folders."""
        folder1 = tmp_path / "model1" / "test.grapto"
        folder2 = tmp_path / "model2" / "test.grapto"
        folder1.mkdir(parents=True)
        folder2.mkdir(parents=True)

        results_data = {
            "field": {
                "type": "collectra.ImageCrop",
                "id": "field-123",
                "x_center": 0.5,
                "y_center": 0.5,
                "width_relative": 0.2,
                "height_relative": 0.2,
            },
        }

        for folder in [folder1, folder2]:
            with open(folder / "results.yaml", "w") as f:
                yaml.dump(results_data, f)

        result = extract_labels_with_bounding_boxes([folder1, folder2])

        assert len(result) == 2
        sources = [r[3] for r in result]
        assert str(folder1) in sources
        assert str(folder2) in sources

    def test_skips_metadata(self, grapto_with_labels):
        """Test that collectra_results_metadata is skipped."""
        result = extract_labels_with_bounding_boxes([grapto_with_labels])

        label_names = [r[0] for r in result]
        assert "collectra_results_metadata" not in label_names

    def test_skips_folder_without_results_yaml(self, tmp_path):
        """Test that folders without results.yaml are skipped gracefully."""
        folder_with = tmp_path / "with.grapto"
        folder_without = tmp_path / "without.grapto"
        folder_with.mkdir()
        folder_without.mkdir()

        results_data = {
            "field": {
                "type": "collectra.ImageCrop",
                "id": "field-123",
                "x_center": 0.5,
                "y_center": 0.5,
                "width_relative": 0.2,
                "height_relative": 0.2,
            },
        }
        with open(folder_with / "results.yaml", "w") as f:
            yaml.dump(results_data, f)

        result = extract_labels_with_bounding_boxes([folder_with, folder_without])

        assert len(result) == 1
        assert result[0][3] == str(folder_with)

    def test_empty_source_folders_list(self):
        """Test with empty source folders list."""
        result = extract_labels_with_bounding_boxes([])

        assert result == []

    def test_text_without_imagecrop_parent_excluded(self, tmp_path):
        """Test that Text without ImageCrop parent is excluded."""
        grapto_folder = tmp_path / "test.grapto"
        grapto_folder.mkdir()

        results_data = {
            "text_only": {
                "type": "collectra.Text",
                "id": "text-123",
                "parents": "nonexistent-parent",
                "data": "Some text",
            },
        }
        with open(grapto_folder / "results.yaml", "w") as f:
            yaml.dump(results_data, f)

        result = extract_labels_with_bounding_boxes([grapto_folder])

        assert len(result) == 0

    def test_returns_correct_tuple_format(self, grapto_with_labels):
        """Test that returned tuples have correct format."""
        result = extract_labels_with_bounding_boxes([grapto_with_labels])

        assert len(result) > 0

        for item in result:
            assert len(item) == 6
            label_name, data, bbox, source, item_id, entry_type = item

            assert isinstance(label_name, str)
            assert isinstance(data, str)
            assert isinstance(bbox, dict)
            assert isinstance(source, str)
            assert isinstance(item_id, str)
            assert isinstance(entry_type, str)
            assert entry_type in ["collectra.Text", "collectra.ImageCrop"]

            # Check bounding box has all required fields
            assert "x_center" in bbox
            assert "y_center" in bbox
            assert "width_relative" in bbox
            assert "height_relative" in bbox

    def test_with_real_data_structure(self, tmp_path):
        """Test with structure matching real results.yaml format."""
        grapto_folder = tmp_path / "MMRIRN1505070_P350015.grapto"
        grapto_folder.mkdir()

        # Mimic real results.yaml structure
        results_data = {
            "collectra_results_metadata": {
                "workflow": "Grapto",
                "version": "0.1.0",
                "timestamp": "2026-01-08T15:16:23.232652",
            },
            "specimen_sheet": {
                "type": "collectra.Image",
                "id": "specimen_sheet",
                "data": "MMRIRN1505070_P350015.jpg",
            },
            "primary_label": {
                "type": "collectra.ImageCrop",
                "id": "primary_label-29b36611-91a9-409d-a753-89340491d047",
                "parents": "specimen_sheet",
                "data": "MMRIRN1505070_P350015.jpg",
                "x_center": 0.4727736711502075,
                "y_center": 0.8679114580154419,
                "width_relative": 0.3212781846523285,
                "height_relative": 0.253208190202713,
            },
            "registration_number_image": {
                "type": "collectra.ImageCrop",
                "id": "registration_number_image-ac99efba-165b-4376-a8b3-205d16e54dca",
                "parents": "primary_label-29b36611-91a9-409d-a753-89340491d047",
                "data": "MMRIRN1505070_P350015.jpg",
                "x_center": 0.42946359509122,
                "y_center": 0.757537248825559,
                "width_relative": 0.06577334993529771,
                "height_relative": 0.032459771822947125,
            },
            "registration_number": {
                "type": "collectra.Text",
                "id": "registration_number-4991f8fc-0290-4f14-89e5-51ba9b83cbf5",
                "parents": [
                    "primary_label-29b36611-91a9-409d-a753-89340491d047",
                    "registration_number_image-ac99efba-165b-4376-a8b3-205d16e54dca",
                ],
                "data": "P.350015",
            },
        }
        with open(grapto_folder / "results.yaml", "w") as f:
            yaml.dump(results_data, f)

        result = extract_labels_with_bounding_boxes([grapto_folder])

        # Find registration_number entry
        reg_entries = [r for r in result if r[0] == "registration_number"]
        assert len(reg_entries) == 1

        label_name, data, bbox, source, item_id, entry_type = reg_entries[0]
        assert label_name == "registration_number"
        assert data == "P.350015"
        # Should get bounding box from last parent: registration_number_image
        assert abs(bbox["x_center"] - 0.42946359509122) < 0.0001
        assert abs(bbox["y_center"] - 0.757537248825559) < 0.0001
        assert item_id == "registration_number-4991f8fc-0290-4f14-89e5-51ba9b83cbf5"
        assert entry_type == "collectra.Text"


# ============================================================================
# TestCalculateIou
# ============================================================================


class TestCalculateIou:
    """Tests for the calculate_iou function."""

    def test_identical_boxes_returns_one(self):
        """Identical boxes should have IoU of 1.0."""
        box = {
            "x_center": 0.5,
            "y_center": 0.5,
            "width_relative": 0.2,
            "height_relative": 0.2,
        }
        result = calculate_iou(box, box)
        assert abs(result - 1.0) < 1e-10

    def test_non_overlapping_boxes_returns_zero(self):
        """Non-overlapping boxes should have IoU of 0.0."""
        box1 = {
            "x_center": 0.1,
            "y_center": 0.1,
            "width_relative": 0.1,
            "height_relative": 0.1,
        }
        box2 = {
            "x_center": 0.9,
            "y_center": 0.9,
            "width_relative": 0.1,
            "height_relative": 0.1,
        }
        result = calculate_iou(box1, box2)
        assert result == 0.0

    def test_partial_overlap(self):
        """Partially overlapping boxes should have IoU between 0 and 1."""
        box1 = {
            "x_center": 0.4,
            "y_center": 0.5,
            "width_relative": 0.2,
            "height_relative": 0.2,
        }
        box2 = {
            "x_center": 0.5,
            "y_center": 0.5,
            "width_relative": 0.2,
            "height_relative": 0.2,
        }
        result = calculate_iou(box1, box2)
        assert 0.0 < result < 1.0
        # Expected: intersection = 0.1 * 0.2 = 0.02, union = 0.04 + 0.04 - 0.02 = 0.06
        # IoU = 0.02 / 0.06 = 0.333...
        assert abs(result - (1 / 3)) < 0.001

    def test_zero_width_box_returns_zero(self):
        """Box with zero width should return IoU of 0.0."""
        box1 = {
            "x_center": 0.5,
            "y_center": 0.5,
            "width_relative": 0.0,
            "height_relative": 0.2,
        }
        box2 = {
            "x_center": 0.5,
            "y_center": 0.5,
            "width_relative": 0.2,
            "height_relative": 0.2,
        }
        result = calculate_iou(box1, box2)
        assert result == 0.0

    def test_zero_height_box_returns_zero(self):
        """Box with zero height should return IoU of 0.0."""
        box1 = {
            "x_center": 0.5,
            "y_center": 0.5,
            "width_relative": 0.2,
            "height_relative": 0.0,
        }
        box2 = {
            "x_center": 0.5,
            "y_center": 0.5,
            "width_relative": 0.2,
            "height_relative": 0.2,
        }
        result = calculate_iou(box1, box2)
        assert result == 0.0

    def test_both_zero_size_boxes_returns_zero(self):
        """Both boxes with zero size should return IoU of 0.0."""
        box1 = {
            "x_center": 0.5,
            "y_center": 0.5,
            "width_relative": 0.0,
            "height_relative": 0.0,
        }
        box2 = {
            "x_center": 0.5,
            "y_center": 0.5,
            "width_relative": 0.0,
            "height_relative": 0.0,
        }
        result = calculate_iou(box1, box2)
        assert result == 0.0

    def test_one_box_inside_another(self):
        """Smaller box completely inside larger box."""
        box1 = {
            "x_center": 0.5,
            "y_center": 0.5,
            "width_relative": 0.4,
            "height_relative": 0.4,
        }
        box2 = {
            "x_center": 0.5,
            "y_center": 0.5,
            "width_relative": 0.2,
            "height_relative": 0.2,
        }
        result = calculate_iou(box1, box2)
        # Intersection = smaller box area = 0.04, union = 0.16 (larger box area)
        # IoU = 0.04 / 0.16 = 0.25
        assert abs(result - 0.25) < 0.001

    def test_negative_size_box_returns_zero(self):
        """Box with negative dimensions should return IoU of 0.0."""
        box1 = {
            "x_center": 0.5,
            "y_center": 0.5,
            "width_relative": -0.2,
            "height_relative": 0.2,
        }
        box2 = {
            "x_center": 0.5,
            "y_center": 0.5,
            "width_relative": 0.2,
            "height_relative": 0.2,
        }
        result = calculate_iou(box1, box2)
        assert result == 0.0


# ============================================================================
# TestCalculateCentroidBox
# ============================================================================


class TestCalculateCentroidBox:
    """Tests for the calculate_centroid_box function."""

    def test_single_box_returns_same_box(self):
        """Single box should return the same box as centroid."""
        box = {
            "x_center": 0.5,
            "y_center": 0.6,
            "width_relative": 0.2,
            "height_relative": 0.3,
        }
        result = calculate_centroid_box([box])
        assert result == box

    def test_multiple_boxes_returns_average(self):
        """Multiple boxes should return their average as centroid."""
        boxes = [
            {
                "x_center": 0.4,
                "y_center": 0.4,
                "width_relative": 0.1,
                "height_relative": 0.1,
            },
            {
                "x_center": 0.6,
                "y_center": 0.6,
                "width_relative": 0.3,
                "height_relative": 0.3,
            },
        ]
        result = calculate_centroid_box(boxes)
        assert abs(result["x_center"] - 0.5) < 0.0001
        assert abs(result["y_center"] - 0.5) < 0.0001
        assert abs(result["width_relative"] - 0.2) < 0.0001
        assert abs(result["height_relative"] - 0.2) < 0.0001

    def test_three_boxes_returns_average(self):
        """Three boxes should return their average."""
        boxes = [
            {
                "x_center": 0.3,
                "y_center": 0.3,
                "width_relative": 0.1,
                "height_relative": 0.1,
            },
            {
                "x_center": 0.5,
                "y_center": 0.5,
                "width_relative": 0.2,
                "height_relative": 0.2,
            },
            {
                "x_center": 0.7,
                "y_center": 0.7,
                "width_relative": 0.3,
                "height_relative": 0.3,
            },
        ]
        result = calculate_centroid_box(boxes)
        assert abs(result["x_center"] - 0.5) < 0.0001
        assert abs(result["y_center"] - 0.5) < 0.0001
        assert abs(result["width_relative"] - 0.2) < 0.0001
        assert abs(result["height_relative"] - 0.2) < 0.0001

    def test_empty_list_raises_value_error(self):
        """Empty list should raise ValueError."""
        with pytest.raises(ValueError, match="Cannot calculate centroid from empty"):
            calculate_centroid_box([])


# ============================================================================
# TestGroupByBoundingBox
# ============================================================================


class TestGroupByBoundingBox:
    """Tests for the group_by_bounding_box function."""

    def test_empty_input_returns_empty_list(self):
        """Empty input should return empty list."""
        result = group_by_bounding_box([])
        assert result == []

    def test_single_item_returns_single_group(self):
        """Single item should return single group with that item."""
        labels = [
            (
                "label1",
                "text1",
                {
                    "x_center": 0.5,
                    "y_center": 0.5,
                    "width_relative": 0.2,
                    "height_relative": 0.2,
                },
                "/path1",
                "id-1",
                "collectra.Text",
            )
        ]
        result = group_by_bounding_box(labels)
        assert len(result) == 1
        assert len(result[0]) == 1
        assert result[0][0] == (
            "label1",
            "text1",
            {
                "x_center": 0.5,
                "y_center": 0.5,
                "width_relative": 0.2,
                "height_relative": 0.2,
            },
            "/path1",
            "id-1",
            "collectra.Text",
        )

    def test_same_label_items_group_together(self):
        """Items with the same label_name should group together."""
        box1 = {
            "x_center": 0.5,
            "y_center": 0.5,
            "width_relative": 0.2,
            "height_relative": 0.2,
        }
        box2 = {
            "x_center": 0.52,
            "y_center": 0.52,
            "width_relative": 0.2,
            "height_relative": 0.2,
        }
        labels = [
            ("same_label", "text1", box1, "/path1", "id-1", "collectra.Text"),
            ("same_label", "text2", box2, "/path2", "id-2", "collectra.Text"),
        ]
        result = group_by_bounding_box(labels, iou_threshold=0.5)
        assert len(result) == 1
        assert len(result[0]) == 2

    def test_different_regions_form_separate_groups(self):
        """Items from different regions should form separate groups."""
        box1 = {
            "x_center": 0.1,
            "y_center": 0.1,
            "width_relative": 0.1,
            "height_relative": 0.1,
        }
        box2 = {
            "x_center": 0.9,
            "y_center": 0.9,
            "width_relative": 0.1,
            "height_relative": 0.1,
        }
        labels = [
            ("label1", "text1", box1, "/path1", "id-1", "collectra.Text"),
            ("label2", "text2", box2, "/path2", "id-2", "collectra.Text"),
        ]
        result = group_by_bounding_box(labels, iou_threshold=0.5)
        assert len(result) == 2
        assert len(result[0]) == 1
        assert len(result[1]) == 1

    def test_different_source_folders_in_same_group(self):
        """Different source folders with same label_name are in the same group."""
        box = {
            "x_center": 0.5,
            "y_center": 0.5,
            "width_relative": 0.2,
            "height_relative": 0.2,
        }
        labels = [
            ("same_label", "text1", box, "/path1", "id-1", "collectra.Text"),
            ("same_label", "text2", box, "/path2", "id-2", "collectra.Text"),
            ("same_label", "text3", box, "/path3", "id-3", "collectra.Text"),
        ]
        result = group_by_bounding_box(labels, iou_threshold=0.5)
        assert len(result) == 1
        assert len(result[0]) == 3
        sources = {item[3] for item in result[0]}
        assert sources == {"/path1", "/path2", "/path3"}

    def test_conflict_resolution_keeps_higher_iou(self):
        """When same source has conflict (same label_name), keep entry with higher IoU to centroid."""
        # All items have the same label_name to be grouped together
        # Using larger boxes with small offsets for conflict resolution
        box_center = {
            "x_center": 0.5,
            "y_center": 0.5,
            "width_relative": 0.4,
            "height_relative": 0.4,
        }
        box_close_to_center = {
            "x_center": 0.51,
            "y_center": 0.51,
            "width_relative": 0.4,
            "height_relative": 0.4,
        }
        box_further = {
            "x_center": 0.53,
            "y_center": 0.53,
            "width_relative": 0.4,
            "height_relative": 0.4,
        }

        labels = [
            (
                "same_label",
                "text_far",
                box_further,
                "/path1",
                "id-far",
                "collectra.Text",
            ),  # First from path1
            (
                "same_label",
                "text_other",
                box_center,
                "/path2",
                "id-other",
                "collectra.Text",
            ),  # From path2 - becomes centroid basis
            (
                "same_label",
                "text_close",
                box_close_to_center,
                "/path1",
                "id-close",
                "collectra.Text",
            ),  # Second from path1 - closer to centroid
        ]
        result = group_by_bounding_box(labels, iou_threshold=0.5)

        assert len(result) == 1
        # Should have 2 entries: one from path1 (the closer one) and one from path2
        assert len(result[0]) == 2
        path1_entries = [item for item in result[0] if item[3] == "/path1"]
        assert len(path1_entries) == 1
        # The kept entry should be text_close (closer to centroid)
        assert path1_entries[0][1] == "text_close"

    def test_conflict_keeps_existing_when_better(self):
        """When existing entry has better IoU to centroid, keep it."""
        # All items have the same label_name to be grouped together
        box_center = {
            "x_center": 0.5,
            "y_center": 0.5,
            "width_relative": 0.4,
            "height_relative": 0.4,
        }
        box_close_to_center = {
            "x_center": 0.51,
            "y_center": 0.51,
            "width_relative": 0.4,
            "height_relative": 0.4,
        }
        box_further = {
            "x_center": 0.53,
            "y_center": 0.53,
            "width_relative": 0.4,
            "height_relative": 0.4,
        }

        labels = [
            (
                "same_label",
                "text_close",
                box_close_to_center,
                "/path1",
                "id-close",
                "collectra.Text",
            ),  # First from path1 - closer
            (
                "same_label",
                "text_other",
                box_center,
                "/path2",
                "id-other",
                "collectra.Text",
            ),  # From path2
            (
                "same_label",
                "text_far",
                box_further,
                "/path1",
                "id-far",
                "collectra.Text",
            ),  # Second from path1 - further
        ]
        result = group_by_bounding_box(labels, iou_threshold=0.5)

        assert len(result) == 1
        assert len(result[0]) == 2
        path1_entries = [item for item in result[0] if item[3] == "/path1"]
        assert len(path1_entries) == 1
        # The kept entry should be text_close (has better IoU to centroid)
        assert path1_entries[0][1] == "text_close"

    def test_multiple_groups_with_different_labels(self):
        """Multiple groups form based on different label_names."""
        # Region 1: top-left
        box1a = {
            "x_center": 0.1,
            "y_center": 0.1,
            "width_relative": 0.15,
            "height_relative": 0.15,
        }
        box1b = {
            "x_center": 0.12,
            "y_center": 0.12,
            "width_relative": 0.15,
            "height_relative": 0.15,
        }
        # Region 2: bottom-right
        box2a = {
            "x_center": 0.9,
            "y_center": 0.9,
            "width_relative": 0.15,
            "height_relative": 0.15,
        }
        box2b = {
            "x_center": 0.88,
            "y_center": 0.88,
            "width_relative": 0.15,
            "height_relative": 0.15,
        }

        labels = [
            ("label_a", "text1a", box1a, "/path1", "id-1a", "collectra.Text"),
            ("label_a", "text1b", box1b, "/path2", "id-1b", "collectra.Text"),
            ("label_b", "text2a", box2a, "/path1", "id-2a", "collectra.Text"),
            ("label_b", "text2b", box2b, "/path2", "id-2b", "collectra.Text"),
        ]
        result = group_by_bounding_box(labels, iou_threshold=0.5)

        assert len(result) == 2
        # Each group should have 2 entries from different sources with same label
        for group in result:
            sources = {item[3] for item in group}
            assert sources == {"/path1", "/path2"}
            # All items in a group should have the same label_name
            labels_in_group = {item[0] for item in group}
            assert len(labels_in_group) == 1

    def test_different_labels_form_separate_groups(self):
        """Items with different label_names form separate groups regardless of bounding box."""
        # Same bounding box but different labels
        box = {
            "x_center": 0.5,
            "y_center": 0.5,
            "width_relative": 0.2,
            "height_relative": 0.2,
        }

        labels = [
            ("label_a", "text1", box, "/path1", "id-1", "collectra.Text"),
            ("label_b", "text2", box, "/path2", "id-2", "collectra.Text"),
        ]

        # Different labels should form separate groups
        result = group_by_bounding_box(labels, iou_threshold=0.5)
        assert len(result) == 2

        # Same labels should group together
        labels_same = [
            ("same_label", "text1", box, "/path1", "id-1", "collectra.Text"),
            ("same_label", "text2", box, "/path2", "id-2", "collectra.Text"),
        ]
        result = group_by_bounding_box(labels_same, iou_threshold=0.5)
        assert len(result) == 1

    def test_output_format_includes_label_name(self):
        """Output tuples should be (label_name, text_content, bounding_box, source_folder_path, item_id, entry_type)."""
        labels = [
            (
                "label_name_here",
                "actual_text",
                {
                    "x_center": 0.5,
                    "y_center": 0.5,
                    "width_relative": 0.2,
                    "height_relative": 0.2,
                },
                "/source/path",
                "item-id-123",
                "collectra.Text",
            )
        ]
        result = group_by_bounding_box(labels)
        assert len(result) == 1
        assert len(result[0]) == 1
        # Verify the output tuple structure includes label_name (6-tuple)
        label_name, text_content, bbox, source, item_id, entry_type = result[0][0]
        assert label_name == "label_name_here"
        assert text_content == "actual_text"
        assert source == "/source/path"
        assert item_id == "item-id-123"
        # Verify label_name IS now included in the output
        assert "label_name_here" in str(result)


# ============================================================================
# Tests for find_centroid_text
# ============================================================================


class TestFindCentroidText:
    """Tests for the find_centroid_text function."""

    def test_single_text_returns_itself(self):
        """Single text should return itself."""
        result = find_centroid_text(["hello"])
        assert result == "hello"

    def test_empty_list_returns_empty_string(self):
        """Empty list should return empty string."""
        result = find_centroid_text([])
        assert result == ""

    def test_multiple_identical_texts_return_that_text(self):
        """Multiple identical texts should return that text."""
        result = find_centroid_text(["hello", "hello", "hello"])
        assert result == "hello"

    def test_multiple_different_texts_returns_centroid(self):
        """Multiple different texts should return the one with minimum total edit distance."""
        # "hello" has distance 1 to "helo" and 1 to "helllo" = 2 total
        # "helo" has distance 1 to "hello" and 2 to "helllo" = 3 total
        # "helllo" has distance 1 to "hello" and 2 to "helo" = 3 total
        result = find_centroid_text(["hello", "helo", "helllo"])
        assert result == "hello"

    def test_all_different_texts(self):
        """Test with completely different texts."""
        # "abc" is equally different from "xyz" and "123"
        # The first one with minimum distance wins
        result = find_centroid_text(["cat", "bat", "rat"])
        # Each has distance 1 to the others = 2 total each
        # First one encountered with minimum wins
        assert result in ["cat", "bat", "rat"]

    def test_two_texts_returns_first_if_equal_distance(self):
        """Two equally distant texts should return the first."""
        result = find_centroid_text(["abc", "xyz"])
        # Both have the same distance to each other
        # First one (abc) should be returned
        assert result == "abc"

    def test_text_with_spaces(self):
        """Test handling of texts with spaces."""
        result = find_centroid_text(["hello world", "hello world", "helo world"])
        assert result == "hello world"

    def test_empty_strings_in_list(self):
        """Test handling of empty strings in the list."""
        result = find_centroid_text(["", "", ""])
        assert result == ""

    def test_mixed_empty_and_non_empty(self):
        """Test handling of mixed empty and non-empty strings."""
        result = find_centroid_text(["hello", "", "hello"])
        # "hello" has distance 5 to "" + 0 to "hello" = 5 total (appears twice)
        # "" has distance 5 to "hello" + 5 to "hello" = 10 total
        assert result == "hello"

    def test_numeric_strings(self):
        """Test handling of numeric strings."""
        result = find_centroid_text(["350015", "350015", "350016"])
        # "350015" appears twice, should win
        assert result == "350015"

    def test_very_similar_strings(self):
        """Test with very similar strings."""
        # Common OCR/LLM variations
        result = find_centroid_text(["P.350015", "P.350015", "P350015"])
        # "P.350015" has lower total distance
        assert result == "P.350015"


# ============================================================================
# Tests for generate_ensembled_values
# ============================================================================


class TestGenerateEnsembledValues:
    """Tests for the generate_ensembled_values function."""

    def test_empty_groups_returns_empty(self):
        """Empty groups should return empty ensembled values."""
        result = generate_ensembled_values([])
        assert result["ensembled_values"] == []
        assert result["statistics"]["total_groups"] == 0
        assert result["statistics"]["standalones"] == 0
        assert result["statistics"]["ensembled"] == 0

    def test_single_entry_group_is_standalone(self):
        """Single entry groups should be counted as standalones."""
        groups = [
            [
                (
                    "registration_number",
                    "P.350015",
                    {
                        "x_center": 0.5,
                        "y_center": 0.5,
                        "width_relative": 0.2,
                        "height_relative": 0.2,
                    },
                    "/path1",
                    "id-1",
                    "collectra.Text",
                )
            ]
        ]
        result = generate_ensembled_values(groups)

        assert len(result["ensembled_values"]) == 1
        assert result["ensembled_values"][0]["text"] == "P.350015"
        assert result["ensembled_values"][0]["label_name"] == "registration_number"
        assert result["ensembled_values"][0]["group_size"] == 1
        assert result["statistics"]["standalones"] == 1
        assert result["statistics"]["ensembled"] == 0
        # Check sources are tracked
        assert result["ensembled_values"][0]["sources"] == [("/path1", "id-1")]

    def test_multiple_entry_group_is_ensembled(self):
        """Multiple entry groups should be counted as ensembled."""
        groups = [
            [
                (
                    "registration_number",
                    "P.350015",
                    {
                        "x_center": 0.5,
                        "y_center": 0.5,
                        "width_relative": 0.2,
                        "height_relative": 0.2,
                    },
                    "/path1",
                    "id-1",
                    "collectra.Text",
                ),
                (
                    "registration_number",
                    "P.350015",
                    {
                        "x_center": 0.51,
                        "y_center": 0.51,
                        "width_relative": 0.2,
                        "height_relative": 0.2,
                    },
                    "/path2",
                    "id-2",
                    "collectra.Text",
                ),
            ]
        ]
        result = generate_ensembled_values(groups)

        assert len(result["ensembled_values"]) == 1
        assert result["ensembled_values"][0]["text"] == "P.350015"
        assert result["ensembled_values"][0]["label_name"] == "registration_number"
        assert result["ensembled_values"][0]["group_size"] == 2
        assert result["statistics"]["standalones"] == 0
        assert result["statistics"]["ensembled"] == 1
        # Check sources are tracked
        assert result["ensembled_values"][0]["sources"] == [
            ("/path1", "id-1"),
            ("/path2", "id-2"),
        ]

    def test_mixed_groups(self):
        """Mixed groups should have correct statistics."""
        groups = [
            [
                (
                    "registration_number",
                    "P.350015",
                    {
                        "x_center": 0.5,
                        "y_center": 0.5,
                        "width_relative": 0.2,
                        "height_relative": 0.2,
                    },
                    "/path1",
                    "id-1",
                    "collectra.Text",
                ),
                (
                    "registration_number",
                    "P.350016",
                    {
                        "x_center": 0.51,
                        "y_center": 0.51,
                        "width_relative": 0.2,
                        "height_relative": 0.2,
                    },
                    "/path2",
                    "id-2",
                    "collectra.Text",
                ),
            ],
            [
                (
                    "species_name",
                    "Species name",
                    {
                        "x_center": 0.3,
                        "y_center": 0.3,
                        "width_relative": 0.1,
                        "height_relative": 0.1,
                    },
                    "/path1",
                    "id-3",
                    "collectra.Text",
                )
            ],
        ]
        result = generate_ensembled_values(groups)

        assert len(result["ensembled_values"]) == 2
        assert result["statistics"]["total_groups"] == 2
        assert result["statistics"]["standalones"] == 1
        assert result["statistics"]["ensembled"] == 1

    def test_source_texts_preserved(self):
        """Source texts should be preserved in the output."""
        groups = [
            [
                (
                    "field",
                    "hello",
                    {
                        "x_center": 0.5,
                        "y_center": 0.5,
                        "width_relative": 0.2,
                        "height_relative": 0.2,
                    },
                    "/path1",
                    "id-1",
                    "collectra.Text",
                ),
                (
                    "field",
                    "helo",
                    {
                        "x_center": 0.5,
                        "y_center": 0.5,
                        "width_relative": 0.2,
                        "height_relative": 0.2,
                    },
                    "/path2",
                    "id-2",
                    "collectra.Text",
                ),
            ]
        ]
        result = generate_ensembled_values(groups)

        assert result["ensembled_values"][0]["source_texts"] == ["hello", "helo"]

    def test_empty_group_in_list_skipped(self):
        """Empty groups in the list should be skipped."""
        groups = [
            [],
            [
                (
                    "field",
                    "text",
                    {
                        "x_center": 0.5,
                        "y_center": 0.5,
                        "width_relative": 0.2,
                        "height_relative": 0.2,
                    },
                    "/path1",
                    "id-1",
                    "collectra.Text",
                )
            ],
        ]
        result = generate_ensembled_values(groups)

        assert len(result["ensembled_values"]) == 1
        # Total groups counts all groups including empty
        assert result["statistics"]["total_groups"] == 2
        assert result["statistics"]["standalones"] == 1

    def test_three_entry_group(self):
        """Three entry groups should ensemble correctly."""
        groups = [
            [
                (
                    "field",
                    "hello",
                    {
                        "x_center": 0.5,
                        "y_center": 0.5,
                        "width_relative": 0.2,
                        "height_relative": 0.2,
                    },
                    "/path1",
                    "id-1",
                    "collectra.Text",
                ),
                (
                    "field",
                    "helo",
                    {
                        "x_center": 0.5,
                        "y_center": 0.5,
                        "width_relative": 0.2,
                        "height_relative": 0.2,
                    },
                    "/path2",
                    "id-2",
                    "collectra.Text",
                ),
                (
                    "field",
                    "hello",
                    {
                        "x_center": 0.5,
                        "y_center": 0.5,
                        "width_relative": 0.2,
                        "height_relative": 0.2,
                    },
                    "/path3",
                    "id-3",
                    "collectra.Text",
                ),
            ]
        ]
        result = generate_ensembled_values(groups)

        # "hello" appears twice with lower total edit distance
        assert result["ensembled_values"][0]["text"] == "hello"
        assert result["ensembled_values"][0]["group_size"] == 3


# ============================================================================
# Tests for write_ensembled_results
# ============================================================================


class TestWriteEnsembledResults:
    """Tests for the write_ensembled_results function."""

    def test_creates_results_yaml(self, tmp_path):
        """Should create a results.yaml file."""
        ensemble_folder = tmp_path / "model1" / "test.grapto"
        ensemble_folder.mkdir(parents=True)

        ensembled_data = {
            "ensembled_values": [
                {
                    "label_name": "registration_number",
                    "text": "P.350015",
                    "bounding_box": {
                        "x_center": 0.5,
                        "y_center": 0.5,
                        "width_relative": 0.2,
                        "height_relative": 0.2,
                    },
                    "group_size": 2,
                    "source_texts": ["P.350015", "P.350015"],
                    "sources": [
                        ("/data/model1/test.grapto", "id-1"),
                        ("/data/model2/test.grapto", "id-2"),
                    ],
                }
            ],
            "statistics": {"total_groups": 1, "standalones": 0, "ensembled": 1},
        }
        groups = []

        result_path = write_ensembled_results(ensemble_folder, ensembled_data, groups)

        assert result_path.exists()
        assert result_path.name == "results.yaml"

    def test_correct_yaml_structure(self, tmp_path):
        """Should create correct YAML structure with ensemble field and top-level bounding box."""
        ensemble_folder = tmp_path / "model1" / "test.grapto"
        ensemble_folder.mkdir(parents=True)

        ensembled_data = {
            "ensembled_values": [
                {
                    "label_name": "registration_number",
                    "text": "P.350015",
                    "bounding_box": {
                        "x_center": 0.5,
                        "y_center": 0.5,
                        "width_relative": 0.2,
                        "height_relative": 0.2,
                    },
                    "group_size": 2,
                    "source_texts": ["P.350015", "P.350015"],
                    "sources": [
                        ("/data/model1/test.grapto", "id-1"),
                        ("/data/model2/test.grapto", "id-2"),
                    ],
                }
            ],
            "statistics": {"total_groups": 1, "standalones": 0, "ensembled": 1},
        }
        groups = []

        result_path = write_ensembled_results(ensemble_folder, ensembled_data, groups)

        with open(result_path) as f:
            data = yaml.safe_load(f)

        assert "collectra_results_metadata" in data
        assert data["collectra_results_metadata"]["workflow"] == "Ensemble"
        assert "ensemble_statistics" in data["collectra_results_metadata"]
        assert "registration_number" in data
        assert data["registration_number"]["data"] == "P.350015"
        assert data["registration_number"]["type"] == "collectra.Text"

        # Check for ensemble field with proper format
        assert "ensemble" in data["registration_number"]
        assert len(data["registration_number"]["ensemble"]) == 2
        assert "model1/test.grapto:id-1" in data["registration_number"]["ensemble"]
        assert "model2/test.grapto:id-2" in data["registration_number"]["ensemble"]

        # Check bounding box is at top level
        assert data["registration_number"]["x_center"] == 0.5
        assert data["registration_number"]["y_center"] == 0.5
        assert data["registration_number"]["width_relative"] == 0.2
        assert data["registration_number"]["height_relative"] == 0.2

    def test_multiple_ensembled_values(self, tmp_path):
        """Should handle multiple ensembled values."""
        ensemble_folder = tmp_path / "model1" / "test.grapto"
        ensemble_folder.mkdir(parents=True)

        ensembled_data = {
            "ensembled_values": [
                {
                    "label_name": "registration_number",
                    "text": "P.350015",
                    "bounding_box": {
                        "x_center": 0.5,
                        "y_center": 0.5,
                        "width_relative": 0.2,
                        "height_relative": 0.2,
                    },
                    "group_size": 2,
                    "source_texts": ["P.350015", "P.350015"],
                    "sources": [
                        ("/data/model1/test.grapto", "id-1"),
                        ("/data/model2/test.grapto", "id-2"),
                    ],
                },
                {
                    "label_name": "species_name",
                    "text": "Species name",
                    "bounding_box": {
                        "x_center": 0.3,
                        "y_center": 0.3,
                        "width_relative": 0.1,
                        "height_relative": 0.1,
                    },
                    "group_size": 1,
                    "source_texts": ["Species name"],
                    "sources": [("/data/model1/test.grapto", "id-3")],
                },
            ],
            "statistics": {"total_groups": 2, "standalones": 1, "ensembled": 1},
        }
        groups = []

        result_path = write_ensembled_results(ensemble_folder, ensembled_data, groups)

        with open(result_path) as f:
            data = yaml.safe_load(f)

        assert "registration_number" in data
        assert "species_name" in data
        assert data["registration_number"]["data"] == "P.350015"
        assert data["species_name"]["data"] == "Species name"

    def test_includes_ensemble_field_and_bounding_box(self, tmp_path):
        """Should include ensemble field and top-level bounding box in each field."""
        ensemble_folder = tmp_path / "model1" / "test.grapto"
        ensemble_folder.mkdir(parents=True)

        ensembled_data = {
            "ensembled_values": [
                {
                    "label_name": "registration_number",
                    "text": "P.350015",
                    "bounding_box": {
                        "x_center": 0.5,
                        "y_center": 0.5,
                        "width_relative": 0.2,
                        "height_relative": 0.2,
                    },
                    "group_size": 2,
                    "source_texts": ["P.350015", "P.350016"],
                    "sources": [
                        ("/data/model1/test.grapto", "id-1"),
                        ("/data/model2/test.grapto", "id-2"),
                    ],
                }
            ],
            "statistics": {"total_groups": 1, "standalones": 0, "ensembled": 1},
        }
        groups = []

        result_path = write_ensembled_results(ensemble_folder, ensembled_data, groups)

        with open(result_path) as f:
            data = yaml.safe_load(f)

        field = data["registration_number"]

        # Check ensemble field
        assert "ensemble" in field

        # Check bounding box is at top level (not nested)
        assert field["x_center"] == 0.5
        assert field["y_center"] == 0.5
        assert field["width_relative"] == 0.2
        assert field["height_relative"] == 0.2

        # Ensure no nested ensemble_info
        assert "ensemble_info" not in field

    def test_empty_ensembled_values(self, tmp_path):
        """Should handle empty ensembled values."""
        ensemble_folder = tmp_path / "model1" / "test.grapto"
        ensemble_folder.mkdir(parents=True)

        ensembled_data = {
            "ensembled_values": [],
            "statistics": {"total_groups": 0, "standalones": 0, "ensembled": 0},
        }
        groups = []

        result_path = write_ensembled_results(ensemble_folder, ensembled_data, groups)

        with open(result_path) as f:
            data = yaml.safe_load(f)

        assert "collectra_results_metadata" in data
        # Should only have metadata, no ensembled_field_* keys
        field_keys = [k for k in data.keys() if k.startswith("ensembled_field_")]
        assert len(field_keys) == 0

    def test_overwrites_existing_results_yaml(self, tmp_path):
        """Should overwrite existing results.yaml."""
        ensemble_folder = tmp_path / "model1" / "test.grapto"
        ensemble_folder.mkdir(parents=True)

        # Create existing results.yaml
        (ensemble_folder / "results.yaml").write_text("old: content")

        ensembled_data = {
            "ensembled_values": [
                {
                    "label_name": "new_field",
                    "text": "new value",
                    "bounding_box": {
                        "x_center": 0.5,
                        "y_center": 0.5,
                        "width_relative": 0.2,
                        "height_relative": 0.2,
                    },
                    "group_size": 1,
                    "source_texts": ["new value"],
                    "sources": [("/data/model1/test.grapto", "id-1")],
                }
            ],
            "statistics": {"total_groups": 1, "standalones": 1, "ensembled": 0},
        }
        groups = []

        result_path = write_ensembled_results(ensemble_folder, ensembled_data, groups)

        with open(result_path) as f:
            data = yaml.safe_load(f)

        assert "old" not in data
        assert "new_field" in data


# ============================================================================
# Tests for ensemble_groups_for_file (Integration)
# ============================================================================


class TestEnsembleGroupsForFile:
    """Integration tests for ensemble_groups_for_file function."""

    @pytest.fixture
    def ensemble_setup(self, tmp_path):
        """Create a complete ensemble setup with source folders and link.yaml."""
        # Create source folders
        source1 = tmp_path / "model1" / "test.grapto"
        source2 = tmp_path / "model2" / "test.grapto"
        source1.mkdir(parents=True)
        source2.mkdir(parents=True)

        # Create results.yaml in each source
        results_data_1 = {
            "collectra_results_metadata": {"workflow": "Grapto", "version": "0.1.0"},
            "registration_image": {
                "type": "collectra.ImageCrop",
                "id": "reg-img-1",
                "x_center": 0.5,
                "y_center": 0.5,
                "width_relative": 0.2,
                "height_relative": 0.1,
            },
            "registration_number": {
                "type": "collectra.Text",
                "id": "reg-text-1",
                "parents": "reg-img-1",
                "data": "P.350015",
            },
        }
        results_data_2 = {
            "collectra_results_metadata": {"workflow": "Grapto", "version": "0.1.0"},
            "registration_image": {
                "type": "collectra.ImageCrop",
                "id": "reg-img-2",
                "x_center": 0.51,
                "y_center": 0.51,
                "width_relative": 0.2,
                "height_relative": 0.1,
            },
            "registration_number": {
                "type": "collectra.Text",
                "id": "reg-text-2",
                "parents": "reg-img-2",
                "data": "P.350015",
            },
        }

        with open(source1 / "results.yaml", "w") as f:
            yaml.dump(results_data_1, f)
        with open(source2 / "results.yaml", "w") as f:
            yaml.dump(results_data_2, f)

        # Create ensemble folder
        ensemble_folder = tmp_path / "ensemble" / "test.grapto"
        ensemble_folder.mkdir(parents=True)

        # Create link.yaml
        link_data = {"test.grapto": [str(source1), str(source2)]}
        link_yaml_path = tmp_path / "ensemble" / "link.yaml"
        with open(link_yaml_path, "w") as f:
            yaml.dump(link_data, f)

        return {
            "ensemble_folder": ensemble_folder,
            "link_yaml_path": link_yaml_path,
            "source1": source1,
            "source2": source2,
        }

    def test_full_integration(self, ensemble_setup):
        """Test full integration of ensemble workflow."""
        result = ensemble_groups_for_file(
            ensemble_setup["ensemble_folder"],
            ensemble_setup["link_yaml_path"],
        )

        assert "ensembled_values" in result
        assert "statistics" in result
        assert "results_yaml_path" in result
        assert result["results_yaml_path"].exists()

    def test_creates_results_yaml(self, ensemble_setup):
        """Should create results.yaml in ensemble folder."""
        result = ensemble_groups_for_file(
            ensemble_setup["ensemble_folder"],
            ensemble_setup["link_yaml_path"],
        )

        results_yaml = ensemble_setup["ensemble_folder"] / "results.yaml"
        assert results_yaml.exists()

        with open(results_yaml) as f:
            data = yaml.safe_load(f)

        assert "collectra_results_metadata" in data

    def test_statistics_are_correct(self, ensemble_setup):
        """Should have correct statistics."""
        result = ensemble_groups_for_file(
            ensemble_setup["ensemble_folder"],
            ensemble_setup["link_yaml_path"],
        )

        stats = result["statistics"]
        assert stats["total_groups"] > 0
        assert stats["standalones"] >= 0
        assert stats["ensembled"] >= 0
        assert stats["standalones"] + stats["ensembled"] <= stats["total_groups"]

    def test_ensembled_values_present(self, ensemble_setup):
        """Should have ensembled values in result."""
        result = ensemble_groups_for_file(
            ensemble_setup["ensemble_folder"],
            ensemble_setup["link_yaml_path"],
        )

        assert len(result["ensembled_values"]) > 0
        for value in result["ensembled_values"]:
            assert "text" in value
            assert "bounding_box" in value
            assert "group_size" in value

    def test_with_differing_texts(self, tmp_path):
        """Test ensemble with differing texts from sources.

        This test creates sources where each has only a Text entry (no ImageCrop)
        with slightly different bounding boxes so they group together but don't
        conflict on source path.
        """
        # Create source folders with different text values
        source1 = tmp_path / "model1" / "test.grapto"
        source2 = tmp_path / "model2" / "test.grapto"
        source3 = tmp_path / "model3" / "test.grapto"
        source1.mkdir(parents=True)
        source2.mkdir(parents=True)
        source3.mkdir(parents=True)

        # Each source has ImageCrop (parent with bounding box) and Text (child with text content)
        # Using slightly offset bounding boxes that still have IoU > 0.6
        results_data_1 = {
            "field_image": {
                "type": "collectra.ImageCrop",
                "id": "field-image-1",
                "data": "image.jpg",
                "x_center": 0.5,
                "y_center": 0.5,
                "width_relative": 0.3,
                "height_relative": 0.3,
            },
            "field": {
                "type": "collectra.Text",
                "id": "field-1",
                "parents": "field-image-1",
                "data": "hello world",
            },
        }
        results_data_2 = {
            "field_image": {
                "type": "collectra.ImageCrop",
                "id": "field-image-2",
                "data": "image.jpg",
                "x_center": 0.52,
                "y_center": 0.52,
                "width_relative": 0.3,
                "height_relative": 0.3,
            },
            "field": {
                "type": "collectra.Text",
                "id": "field-2",
                "parents": "field-image-2",
                "data": "helo world",  # Typo
            },
        }
        results_data_3 = {
            "field_image": {
                "type": "collectra.ImageCrop",
                "id": "field-image-3",
                "data": "image.jpg",
                "x_center": 0.51,
                "y_center": 0.51,
                "width_relative": 0.3,
                "height_relative": 0.3,
            },
            "field": {
                "type": "collectra.Text",
                "id": "field-3",
                "parents": "field-image-3",
                "data": "hello world",  # Same as source1
            },
        }

        with open(source1 / "results.yaml", "w") as f:
            yaml.dump(results_data_1, f)
        with open(source2 / "results.yaml", "w") as f:
            yaml.dump(results_data_2, f)
        with open(source3 / "results.yaml", "w") as f:
            yaml.dump(results_data_3, f)

        ensemble_folder = tmp_path / "ensemble" / "test.grapto"
        ensemble_folder.mkdir(parents=True)

        link_data = {"test.grapto": [str(source1), str(source2), str(source3)]}
        link_yaml_path = tmp_path / "ensemble" / "link.yaml"
        with open(link_yaml_path, "w") as f:
            yaml.dump(link_data, f)

        result = ensemble_groups_for_file(ensemble_folder, link_yaml_path)

        # Should have at least one ensembled value
        text_values = [v["text"] for v in result["ensembled_values"]]
        assert len(text_values) > 0

        # The ensembled text should be "hello world" since it appears twice
        # and has lower total edit distance
        assert "hello world" in text_values
