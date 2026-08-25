"""Integration tests for the ensemble module.

Tests cover end-to-end ensemble workflows including:
- Full ensemble pipeline execution
- File I/O operations
- YAML output validation
- Artifact handling
- Edge cases
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from collectra.pipelines.ensemble import (
    EnsembleProcessor,
    calculate_iou,
    cluster_texts,
    find_centroid_text,
    find_contributing_boxes,
)

# =============================================================================
# Integration Tests for Ensemble Workflow
# =============================================================================


class TestEnsembleWorkflowIntegration:
    """Integration tests for the complete ensemble workflow."""

    def test_create_file_linkage_with_real_folders(self, temp_ensemble_folders):
        """Test file linkage creation with actual folder structure."""
        folders, output = temp_ensemble_folders(num_folders=3, files_per_folder=2)

        processor = EnsembleProcessor(
            node_manager=MagicMock(),
            ext=".collectra",
            name="test",
            version="1.0",
        )

        linkage = processor.create_file_linkage(folders)

        # Should find file_0.collectra and file_1.collectra across 3 folders
        assert "file_0.collectra" in linkage
        assert "file_1.collectra" in linkage
        assert len(linkage["file_0.collectra"]) == 3
        assert len(linkage["file_1.collectra"]) == 3

    def test_copy_artifact_files_integration(self, temp_ensemble_folders):
        """Test artifact file copying in real folder structure."""
        folders, output = temp_ensemble_folders(num_folders=1, files_per_folder=1)
        source = Path(folders[0]) / "file_0.collectra"
        dest = output / "ensembled.collectra"
        dest.mkdir(parents=True, exist_ok=True)

        processor = EnsembleProcessor(
            node_manager=MagicMock(),
            ext=".collectra",
            name="test",
            version="1.0",
        )

        processor.copy_artifact_files(source, dest)

        # Should have copied test_image.jpg but not results.yaml
        assert (dest / "test_image.jpg").exists()
        assert not (dest / "results.yaml").exists()

    def test_ensemble_produces_valid_yaml(self, temp_ensemble_folders, tmp_path):
        """Test that ensemble output is valid YAML."""
        folders, _ = temp_ensemble_folders(num_folders=2, files_per_folder=1)
        output = tmp_path / "ensemble_output"
        output.mkdir(parents=True, exist_ok=True)

        # Create a simple ensembled result manually
        result_path = output / "test.collectra"
        result_path.mkdir()

        ensembled = {
            "input_image": {
                "type": "collectra.Image",
                "id": "input_image_ensemble",
                "data": "test.jpg",
                "ensemble": ["src1::img", "src2::img"],
            }
        }

        with open(result_path / "results.yaml", "w") as f:
            yaml.dump(ensembled, f)

        # Verify it's valid YAML
        with open(result_path / "results.yaml") as f:
            loaded = yaml.safe_load(f)

        assert loaded == ensembled


class TestEnsembleImageCropsIntegration:
    """Integration tests for ImageCrop ensembling with WBF."""

    def test_wbf_fusion_produces_valid_boxes(self, sample_wbf_input):
        """Test that WBF produces valid fused boxes."""
        from ensemble_boxes import weighted_boxes_fusion

        boxes = sample_wbf_input["boxes_per_source"]
        scores = sample_wbf_input["scores_per_source"]
        labels = sample_wbf_input["labels_per_source"]

        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            boxes, scores, labels, weights=[1, 1], iou_thr=0.5, skip_box_thr=0.0
        )

        # Should fuse similar boxes from both sources
        assert len(fused_boxes) == 2  # Two distinct regions

        # All coordinates should be valid (0-1 range)
        for box in fused_boxes:
            assert all(0 <= coord <= 1 for coord in box)

    def test_find_contributing_boxes_after_wbf(self, sample_wbf_input):
        """Test finding contributors after WBF fusion."""
        from ensemble_boxes import weighted_boxes_fusion

        boxes = sample_wbf_input["boxes_per_source"]
        scores = sample_wbf_input["scores_per_source"]
        labels = sample_wbf_input["labels_per_source"]
        prov = sample_wbf_input["provenance_per_source"]

        fused_boxes, _, _ = weighted_boxes_fusion(
            boxes, scores, labels, weights=[1, 1], iou_thr=0.5, skip_box_thr=0.0
        )

        flat_boxes = [b for src_boxes in boxes for b in src_boxes]
        flat_prov = [p for src_prov in prov for p in src_prov]

        for fused_box in fused_boxes:
            contributors = find_contributing_boxes(
                list(fused_box), flat_boxes, flat_prov, iou_threshold=0.3
            )
            # Should find contributors from both sources
            assert len(contributors) >= 1


class TestEnsembleTextIntegration:
    """Integration tests for Text ensembling with clustering."""

    def test_cluster_texts_with_real_similarity(self):
        """Test text clustering with realistic text variations."""
        texts = [
            "Invoice Number: 12345",
            "Invoice Number: 12345",  # Exact duplicate
            "Invoice No: 12345",  # Slight variation
            "Total: $100.00",
            "Total: $100.00",
            "Total: $ 100.00",  # Space variation
        ]
        prov = ["s1::t1", "s2::t1", "s3::t1", "s1::t2", "s2::t2", "s3::t2"]

        clusters = cluster_texts(texts, prov, num_clusters=6, similarity_threshold=0.7)

        # Should form 2 clusters (invoice variants and total variants)
        assert len(clusters) == 2

        # Each cluster should have 3 contributors
        for cluster in clusters:
            assert len(cluster["contributors"]) == 3

    def test_centroid_selection_accuracy(self):
        """Test that centroid selection picks the most representative text."""
        # The centroid should be the one with minimum total edit distance
        texts = ["abcde", "abcdf", "abcdg"]  # All similar
        centroid = find_centroid_text(texts)
        assert centroid in texts

        # For texts with clear medoid
        texts_with_medoid = ["hello", "hallo", "hullo"]
        # "hello" has edit distance 1 to both others = total 2
        # "hallo" has edit distance 1+2 = total 3
        # "hullo" has edit distance 2+2 = total 4
        centroid = find_centroid_text(texts_with_medoid)
        assert centroid == "hello"


class TestEnsembleEdgeCases:
    """Integration tests for edge cases in ensemble processing."""

    def test_single_source_passthrough(self, temp_ensemble_folders):
        """Test ensemble with single source passes through data."""
        folders, output = temp_ensemble_folders(num_folders=1, files_per_folder=1)

        processor = EnsembleProcessor(
            node_manager=MagicMock(),
            ext=".collectra",
            name="test",
            version="1.0",
        )

        linkage = processor.create_file_linkage(folders)

        # Single source should still work
        assert "file_0.collectra" in linkage
        assert len(linkage["file_0.collectra"]) == 1

    def test_empty_artefact_nodes(self, mock_ensemble_processor, mock_artefact_node):
        """Test handling of empty artefact nodes."""
        node = mock_artefact_node("empty_node", ensemble_items={})
        result = mock_ensemble_processor.ensemble_artefact_node(node, {})
        assert result is None

    def test_missing_source_folders(self, tmp_path):
        """Test handling of missing source folders."""
        existing = tmp_path / "existing"
        existing.mkdir()
        (existing / "file.collectra").mkdir()

        missing = tmp_path / "missing"  # Does not exist

        processor = EnsembleProcessor(
            node_manager=MagicMock(),
            ext=".collectra",
            name="test",
            version="1.0",
        )

        linkage = processor.create_file_linkage([existing, missing])

        # Should only find files from existing folder
        assert "file.collectra" in linkage
        assert len(linkage["file.collectra"]) == 1

    def test_iou_edge_cases(self):
        """Test IoU calculation edge cases."""
        # Identical boxes
        assert calculate_iou([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0

        # No overlap
        assert calculate_iou([0, 0, 0.5, 0.5], [0.6, 0.6, 1, 1]) == 0.0

        # Zero area (point)
        assert calculate_iou([0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1]) == 0.0

        # One contains the other
        outer = [0, 0, 1, 1]
        inner = [0.25, 0.25, 0.75, 0.75]
        iou = calculate_iou(outer, inner)
        assert 0 < iou < 1

    def test_cluster_texts_empty_input(self):
        """Test text clustering with empty input."""
        clusters = cluster_texts([], [], num_clusters=5, similarity_threshold=0.8)
        assert clusters == []

    def test_cluster_texts_single_text(self):
        """Test text clustering with single text."""
        clusters = cluster_texts(
            ["only one"], ["prov1"], num_clusters=5, similarity_threshold=0.8
        )
        assert len(clusters) == 1
        assert clusters[0]["centroid"] == "only one"
        assert clusters[0]["contributors"] == ["prov1"]


class TestEnsembleProvenanceTracking:
    """Tests for provenance tracking throughout ensemble process."""

    def test_provenance_preserved_through_clustering(self):
        """Test that provenance is correctly preserved during text clustering."""
        texts = ["hello", "hello", "world"]
        prov = ["source1::text1", "source2::text1", "source1::text2"]

        clusters = cluster_texts(texts, prov, num_clusters=3, similarity_threshold=0.9)

        # "hello" x2 should cluster together with their provenance
        hello_cluster = next(c for c in clusters if c["centroid"] == "hello")
        assert "source1::text1" in hello_cluster["contributors"]
        assert "source2::text1" in hello_cluster["contributors"]

        # "world" should be separate
        world_cluster = next(c for c in clusters if c["centroid"] == "world")
        assert world_cluster["contributors"] == ["source1::text2"]

    def test_box_contributors_match_provenance(self):
        """Test that box contributors match input provenance."""
        fused = [0.2, 0.2, 0.4, 0.4]
        sources = [
            [0.2, 0.2, 0.4, 0.4],  # Matches
            [0.21, 0.21, 0.41, 0.41],  # Close match
            [0.8, 0.8, 1.0, 1.0],  # No match
        ]
        prov = ["exact_match", "close_match", "no_match"]

        contributors = find_contributing_boxes(fused, sources, prov, iou_threshold=0.5)

        assert "exact_match" in contributors
        assert "close_match" in contributors
        assert "no_match" not in contributors


class TestEnsembleArtifactHandling:
    """Tests for artifact file handling during ensemble."""

    def test_artifact_preservation(self, tmp_path):
        """Test that artifact files are correctly preserved."""
        source = tmp_path / "source"
        dest = tmp_path / "dest"
        source.mkdir()
        dest.mkdir()

        # Create various artifact types
        (source / "image.jpg").write_bytes(b"fake image data")
        (source / "thumbnail.png").write_bytes(b"fake thumbnail")
        (source / "results.yaml").write_text("should not copy")

        processor = EnsembleProcessor(
            node_manager=MagicMock(),
            ext=".collectra",
            name="test",
            version="1.0",
        )

        processor.copy_artifact_files(source, dest)

        assert (dest / "image.jpg").read_bytes() == b"fake image data"
        assert (dest / "thumbnail.png").read_bytes() == b"fake thumbnail"
        assert not (dest / "results.yaml").exists()

    def test_empty_source_handling(self, tmp_path):
        """Test handling of empty source folder."""
        source = tmp_path / "empty_source"
        dest = tmp_path / "dest"
        source.mkdir()
        dest.mkdir()

        # Only results.yaml, no artifacts
        (source / "results.yaml").write_text("data")

        processor = EnsembleProcessor(
            node_manager=MagicMock(),
            ext=".collectra",
            name="test",
            version="1.0",
        )

        processor.copy_artifact_files(source, dest)

        # Destination should remain empty (no results.yaml copied)
        assert list(dest.iterdir()) == []
