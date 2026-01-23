"""Unit tests for the ensemble module.

Tests cover:
- Pure utility functions (calculate_iou, find_centroid_text, cluster_texts, find_contributing_boxes)
- EnsembleProcessor class methods
"""

from unittest.mock import MagicMock, patch

import pytest

from collectra.pipelines.ensemble import (
    EnsembleProcessor,
    calculate_iou,
    cluster_texts,
    find_centroid_text,
    find_contributing_boxes,
)

# =============================================================================
# Tests for calculate_iou
# =============================================================================


class TestCalculateIou:
    """Tests for the calculate_iou function."""

    def test_identical_boxes_returns_one(self, sample_boxes):
        """Identical boxes should have IoU of 1.0."""
        result = calculate_iou(sample_boxes["box1"], sample_boxes["box4"])
        assert result == 1.0

    def test_non_overlapping_boxes_returns_zero(self, sample_boxes):
        """Non-overlapping boxes should have IoU of 0.0."""
        result = calculate_iou(
            sample_boxes["box1"], sample_boxes["box_non_overlapping"]
        )
        assert result == 0.0

    def test_partial_overlap_returns_expected_value(self, sample_boxes):
        """Partially overlapping boxes should return correct IoU."""
        # box1: [0, 0, 0.5, 0.5] area = 0.25
        # box2: [0.25, 0.25, 0.75, 0.75] area = 0.25
        # intersection: [0.25, 0.25, 0.5, 0.5] area = 0.0625
        # union: 0.25 + 0.25 - 0.0625 = 0.4375
        # iou: 0.0625 / 0.4375 = 0.142857...
        result = calculate_iou(sample_boxes["box1"], sample_boxes["box2"])
        assert 0.14 < result < 0.15

    def test_zero_area_box_returns_zero(self, sample_boxes):
        """Zero area box should return IoU of 0.0."""
        result = calculate_iou(sample_boxes["box1"], sample_boxes["box_zero_area"])
        assert result == 0.0

    def test_containing_box(self):
        """Box fully contained in another should have partial IoU."""
        outer = [0.0, 0.0, 1.0, 1.0]  # Full image
        inner = [0.25, 0.25, 0.75, 0.75]  # Center quarter
        # inner area: 0.25, outer area: 1.0
        # intersection: 0.25, union: 1.0
        # iou: 0.25
        result = calculate_iou(outer, inner)
        assert result == 0.25

    def test_adjacent_boxes_zero_overlap(self):
        """Adjacent boxes (touching but not overlapping) should have IoU of 0."""
        box1 = [0.0, 0.0, 0.5, 0.5]
        box2 = [0.5, 0.0, 1.0, 0.5]  # Right of box1
        result = calculate_iou(box1, box2)
        assert result == 0.0


# =============================================================================
# Tests for find_centroid_text
# =============================================================================


class TestFindCentroidText:
    """Tests for the find_centroid_text function."""

    def test_single_text_returns_same(self, sample_texts):
        """Single text should return itself."""
        result = find_centroid_text(sample_texts["single"])
        assert result == "only one"

    def test_finds_medoid_correctly(self, sample_texts):
        """Should find the text with minimum total edit distance."""
        # "hello world" is most similar to "hello worlds" and "helo world"
        result = find_centroid_text(sample_texts["similar"])
        assert result == "hello world"

    def test_identical_texts_returns_first(self, sample_texts):
        """Identical texts should return the first one (any would be valid)."""
        result = find_centroid_text(sample_texts["identical"])
        assert result == "hello world"

    def test_distinct_texts_returns_valid_medoid(self, sample_texts):
        """For distinct texts, should return a valid medoid."""
        result = find_centroid_text(sample_texts["distinct"])
        # The medoid should be one of the input texts
        assert result in sample_texts["distinct"]


# =============================================================================
# Tests for cluster_texts
# =============================================================================


class TestClusterTexts:
    """Tests for the cluster_texts function."""

    def test_identical_texts_single_cluster(self, sample_texts, sample_provenance):
        """Identical texts should form a single cluster."""
        result = cluster_texts(
            sample_texts["identical"],
            sample_provenance["identical"],
            num_clusters=3,
            similarity_threshold=0.8,
        )
        assert len(result) == 1
        assert result[0]["centroid"] == "hello world"
        assert len(result[0]["contributors"]) == 3

    def test_distinct_texts_separate_clusters(self, sample_texts, sample_provenance):
        """Distinct texts should form separate clusters."""
        result = cluster_texts(
            sample_texts["distinct"],
            sample_provenance["distinct"],
            num_clusters=3,
            similarity_threshold=0.8,
        )
        # Each distinct text forms its own cluster
        assert len(result) == 3
        centroids = [c["centroid"] for c in result]
        assert "apple" in centroids
        assert "banana" in centroids
        assert "cherry" in centroids

    def test_provenance_correctly_tracked(self, sample_texts, sample_provenance):
        """Provenance should be correctly tracked for each cluster."""
        result = cluster_texts(
            sample_texts["similar"],
            sample_provenance["similar"],
            num_clusters=3,
            similarity_threshold=0.7,
        )
        # All similar texts should cluster together
        assert len(result) == 1
        assert set(result[0]["contributors"]) == set(sample_provenance["similar"])

    def test_single_linkage_transitive(self, sample_texts, sample_provenance):
        """Single linkage should allow transitive clustering."""
        # "hello" -> "hello!" -> but not "goodbye"
        result = cluster_texts(
            sample_texts["mixed"],
            sample_provenance["mixed"],
            num_clusters=4,
            similarity_threshold=0.7,
        )
        # Should have 2 clusters: hello variants and goodbye variants
        assert len(result) == 2

    def test_num_clusters_respected(self, sample_texts, sample_provenance):
        """Should not create more clusters than num_clusters."""
        result = cluster_texts(
            sample_texts["distinct"],
            sample_provenance["distinct"],
            num_clusters=2,  # Limit to 2 clusters
            similarity_threshold=0.9,
        )
        assert len(result) <= 2

    def test_empty_input_returns_empty(self, sample_texts, sample_provenance):
        """Empty input should return empty list."""
        result = cluster_texts(
            sample_texts["empty"],
            sample_provenance["empty"],
            num_clusters=3,
            similarity_threshold=0.8,
        )
        assert result == []


# =============================================================================
# Tests for find_contributing_boxes
# =============================================================================


class TestFindContributingBoxes:
    """Tests for the find_contributing_boxes function."""

    def test_exact_match_found(self, sample_boxes):
        """Exact matching box should be found."""
        fused = sample_boxes["box1"]
        sources = [sample_boxes["box1"], sample_boxes["box3"]]
        prov = ["source1", "source2"]
        result = find_contributing_boxes(fused, sources, prov, iou_threshold=0.5)
        assert "source1" in result
        assert "source2" not in result

    def test_overlapping_boxes_above_threshold(self, sample_boxes):
        """Overlapping boxes above threshold should be found."""
        fused = [0.1, 0.1, 0.4, 0.4]
        source1 = [0.1, 0.1, 0.4, 0.4]  # Identical
        source2 = [0.15, 0.15, 0.45, 0.45]  # Overlapping
        sources = [source1, source2]
        prov = ["src1", "src2"]
        result = find_contributing_boxes(fused, sources, prov, iou_threshold=0.5)
        assert "src1" in result
        assert "src2" in result

    def test_non_overlapping_filtered_out(self, sample_boxes):
        """Non-overlapping boxes should be filtered out."""
        fused = sample_boxes["box1"]
        sources = [sample_boxes["box1"], sample_boxes["box_non_overlapping"]]
        prov = ["good", "bad"]
        result = find_contributing_boxes(fused, sources, prov, iou_threshold=0.5)
        assert "good" in result
        assert "bad" not in result

    def test_raises_when_no_contributors(self, sample_boxes):
        """Should raise ValueError when no contributors found."""
        fused = sample_boxes["box1"]
        sources = [sample_boxes["box_non_overlapping"]]
        prov = ["no_match"]
        with pytest.raises(ValueError, match="No contributing boxes found"):
            find_contributing_boxes(fused, sources, prov, iou_threshold=0.5)

    def test_multiple_contributors(self):
        """Should find all contributing boxes above threshold."""
        fused = [0.2, 0.2, 0.6, 0.6]
        sources = [
            [0.2, 0.2, 0.6, 0.6],  # Exact match
            [0.21, 0.21, 0.61, 0.61],  # Very close
            [0.22, 0.22, 0.62, 0.62],  # Close
            [0.9, 0.9, 1.0, 1.0],  # No overlap
        ]
        prov = ["exact", "close1", "close2", "far"]
        result = find_contributing_boxes(fused, sources, prov, iou_threshold=0.5)
        assert "exact" in result
        assert "close1" in result
        assert "close2" in result
        assert "far" not in result


# =============================================================================
# Tests for EnsembleProcessor.create_file_linkage
# =============================================================================


class TestCreateFileLinkage:
    """Tests for EnsembleProcessor.create_file_linkage method."""

    def test_single_folder_single_file(self, mock_ensemble_processor, tmp_path):
        """Single folder with single file should create correct linkage."""
        folder = tmp_path / "source"
        folder.mkdir()
        (folder / "file1.collectra").mkdir()

        result = mock_ensemble_processor.create_file_linkage([folder])

        assert "file1.collectra" in result
        assert len(result["file1.collectra"]) == 1

    def test_multiple_folders_same_file(self, mock_ensemble_processor, tmp_path):
        """Same file in multiple folders should link together."""
        folder1 = tmp_path / "source1"
        folder2 = tmp_path / "source2"
        folder1.mkdir()
        folder2.mkdir()
        (folder1 / "common.collectra").mkdir()
        (folder2 / "common.collectra").mkdir()

        result = mock_ensemble_processor.create_file_linkage([folder1, folder2])

        assert "common.collectra" in result
        assert len(result["common.collectra"]) == 2

    def test_empty_folder_skipped(self, mock_ensemble_processor, tmp_path):
        """Empty folders should be skipped."""
        folder1 = tmp_path / "source1"
        folder2 = tmp_path / "source2"  # Will not exist
        folder1.mkdir()
        (folder1 / "file.collectra").mkdir()

        result = mock_ensemble_processor.create_file_linkage([folder1, folder2])

        assert "file.collectra" in result
        assert len(result["file.collectra"]) == 1

    def test_non_matching_extension_ignored(self, mock_ensemble_processor, tmp_path):
        """Files/folders without matching extension should be ignored."""
        folder = tmp_path / "source"
        folder.mkdir()
        (folder / "file1.collectra").mkdir()
        (folder / "file2.other").mkdir()  # Wrong extension

        result = mock_ensemble_processor.create_file_linkage([folder])

        assert "file1.collectra" in result
        assert "file2.other" not in result


# =============================================================================
# Tests for EnsembleProcessor.ensemble_data_node
# =============================================================================


class TestEnsembleDataNode:
    """Tests for EnsembleProcessor.ensemble_data_node method."""

    def test_empty_items_returns_none(self, mock_ensemble_processor, mock_data_node):
        """Empty ensemble_items should return None."""
        node = mock_data_node("test_node", ensemble_items={})
        result = mock_ensemble_processor.ensemble_data_node(node, {})
        assert result is None

    def test_image_returns_first_with_ensemble_refs(
        self, mock_ensemble_processor, mock_data_node
    ):
        """Image type should return first item with ensemble refs."""
        from collectra.types.images import Image

        # Create mock images
        img1 = MagicMock(spec=Image)
        img1.get_path.return_value.name = "test.jpg"
        img2 = MagicMock(spec=Image)
        img2.get_path.return_value.name = "test.jpg"

        # Mock type checking
        type(img1).__name__ = "Image"
        type(img2).__name__ = "Image"

        node = mock_data_node(
            "input_image",
            ensemble_items={
                "src1::img1": img1,
                "src2::img1": img2,
            },
        )

        # Patch type() to return Image
        with patch("collectra.pipelines.ensemble.type") as mock_type:
            mock_type.return_value = Image
            result = mock_ensemble_processor.ensemble_data_node(node, {})

        assert result["type"] == "collectra.Image"
        assert result["id"] == "input_image_ensemble"
        assert "ensemble" in result

    def test_image_crops_calls_ensemble_method(
        self, mock_ensemble_processor, mock_data_node, mock_image_crop
    ):
        """ImageCrop type should call ensemble_image_crops method."""
        from collectra.types.images import ImageCrop

        crop1 = mock_image_crop("src1::crop1")
        crop2 = mock_image_crop("src2::crop1")

        node = mock_data_node(
            "crops",
            ensemble_items={
                "src1::crop1": crop1,
                "src2::crop1": crop2,
            },
        )

        # Mock the ensemble_image_crops method
        mock_ensemble_processor.ensemble_image_crops = MagicMock(
            return_value={"mocked": True}
        )

        with patch("collectra.pipelines.ensemble.type") as mock_type:
            mock_type.return_value = ImageCrop
            result = mock_ensemble_processor.ensemble_data_node(node, {})

        mock_ensemble_processor.ensemble_image_crops.assert_called_once()

    def test_text_calls_ensemble_method(
        self, mock_ensemble_processor, mock_data_node, mock_text
    ):
        """Text type should call ensemble_text method."""
        from collectra.types.texts import Text

        text1 = mock_text("src1::text1", "hello")
        text2 = mock_text("src2::text1", "hello")

        node = mock_data_node(
            "texts",
            ensemble_items={
                "src1::text1": text1,
                "src2::text1": text2,
            },
        )

        # Mock the ensemble_text method
        mock_ensemble_processor.ensemble_text = MagicMock(return_value={"mocked": True})

        with patch("collectra.pipelines.ensemble.type") as mock_type:
            mock_type.return_value = Text
            result = mock_ensemble_processor.ensemble_data_node(node, {})

        mock_ensemble_processor.ensemble_text.assert_called_once()


# =============================================================================
# Tests for EnsembleProcessor.resolve_ensemble_parents
# =============================================================================


class TestResolveEnsembleParents:
    """Tests for EnsembleProcessor.resolve_ensemble_parents method."""

    def test_single_parent_returns_string(
        self, mock_ensemble_processor, sample_ensemble_data
    ):
        """Single parent should return as string, not list."""
        # Create mock nodes with parents
        node = MagicMock()
        node.parents = ["src1::crop1"]

        result = mock_ensemble_processor.resolve_ensemble_parents(
            [node], sample_ensemble_data
        )

        assert result == "detected_region_ensembled_1"

    def test_multiple_parents_returns_list(
        self, mock_ensemble_processor, sample_ensemble_data
    ):
        """Multiple parents should return as list."""
        node1 = MagicMock()
        node1.parents = ["src1::crop1"]
        node2 = MagicMock()
        node2.parents = ["src1::crop2"]

        result = mock_ensemble_processor.resolve_ensemble_parents(
            [node1, node2], sample_ensemble_data
        )

        assert isinstance(result, list)
        assert "detected_region_ensembled_1" in result
        assert "detected_region_ensembled_2" in result

    def test_parent_from_dict_ensemble(self, mock_ensemble_processor):
        """Should resolve parent from dict-type ensemble data."""
        ensemble_data = {
            "input": {
                "id": "input_ensemble",
                "ensemble": ["src1::input", "src2::input"],
            }
        }
        node = MagicMock()
        node.parents = ["src1::input"]

        result = mock_ensemble_processor.resolve_ensemble_parents([node], ensemble_data)

        assert result == "input_ensemble"

    def test_no_matching_parent_returns_empty(self, mock_ensemble_processor):
        """No matching parent should return empty list."""
        ensemble_data = {
            "input": {
                "id": "input_ensemble",
                "ensemble": ["src1::input", "src2::input"],
            }
        }
        node = MagicMock()
        node.parents = ["nonexistent"]

        result = mock_ensemble_processor.resolve_ensemble_parents([node], ensemble_data)

        assert result == []


# =============================================================================
# Tests for EnsembleProcessor.copy_artifact_files
# =============================================================================


class TestCopyArtifactFiles:
    """Tests for EnsembleProcessor.copy_artifact_files method."""

    def test_copies_non_yaml_files(self, mock_ensemble_processor, tmp_path):
        """Should copy all files except results.yaml."""
        source = tmp_path / "source"
        dest = tmp_path / "dest"
        source.mkdir()
        dest.mkdir()

        (source / "image.jpg").touch()
        (source / "results.yaml").touch()

        mock_ensemble_processor.copy_artifact_files(source, dest)

        assert (dest / "image.jpg").exists()
        assert not (dest / "results.yaml").exists()

    def test_skips_nonexistent_source(self, mock_ensemble_processor, tmp_path):
        """Should skip if source doesn't exist."""
        source = tmp_path / "nonexistent"
        dest = tmp_path / "dest"
        dest.mkdir()

        # Should not raise
        mock_ensemble_processor.copy_artifact_files(source, dest)

    def test_skips_directories(self, mock_ensemble_processor, tmp_path):
        """Should only copy files, not directories."""
        source = tmp_path / "source"
        dest = tmp_path / "dest"
        source.mkdir()
        dest.mkdir()

        (source / "subdir").mkdir()
        (source / "file.txt").touch()

        mock_ensemble_processor.copy_artifact_files(source, dest)

        assert (dest / "file.txt").exists()
        assert not (dest / "subdir").exists()
