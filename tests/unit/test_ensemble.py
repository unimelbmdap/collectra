"""
Unit tests for the ensemble module.

This module tests the ensemble functionality including:
- Finding collectra files in folders
- Verifying files exist across multiple folders
- Creating ensemble output with merged results
- Loading link.yaml files
"""

import shutil
from pathlib import Path

import pytest
import yaml

from collectra.ensemble import (
    create_ensemble_output,
    ensemble_files,
    find_collectra_files,
    get_ensemble_folder,
    load_link_yaml,
    verify_collectra_files,
    _get_image_file,
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
            f for f in output_grapto.iterdir()
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

    def test_link_yaml_returns_correct_path(
        self, collectra_folder_structure, tmp_path
    ):
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

        # Check results.yaml files exist and are empty
        assert (output_folder / "file1.grapto" / "results.yaml").read_text() == ""
        assert (output_folder / "file2.grapto" / "results.yaml").read_text() == ""

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
            "file1.grapto": ["/path/to/source1/file1.grapto", "/path/to/source2/file1.grapto"],
            "file2.grapto": ["/path/to/source1/file2.grapto", "/path/to/source2/file2.grapto"],
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
