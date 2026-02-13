"""
Tests for collectra.converters module — Converter ABC, YOLOConfig, YOLOConverter, convert_files.
"""

import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from collectra.converters import convert_files
from collectra.converters.base import Converter
from collectra.converters.yolo import YOLOConfig, YOLOConverter


# ---------------------------------------------------------------------------
# Converter ABC
# ---------------------------------------------------------------------------
class TestConverterABC:
    """Tests for the Converter abstract base class."""

    def test_cannot_instantiate_directly(self):
        """Abstract class should not be instantiable."""
        with pytest.raises(TypeError):
            Converter()

    def test_concrete_subclass_works(self):
        """A concrete subclass implementing convert() should be instantiable."""

        class MyConverter(Converter):
            def convert(self):
                return "converted"

        c = MyConverter()
        assert c.convert() == "converted"


# ---------------------------------------------------------------------------
# YOLOConfig
# ---------------------------------------------------------------------------
class TestYOLOConfig:
    """Tests for the YOLOConfig pydantic model."""

    def test_valid_config(self):
        """Should construct successfully with all required fields."""
        config = YOLOConfig(
            train="train.txt", val="val.txt", nc=3, names=["a", "b", "c"]
        )
        assert config.train == "train.txt"
        assert config.nc == 3
        assert len(config.names) == 3

    def test_missing_field_raises(self):
        """Missing a required field should raise a validation error."""
        with pytest.raises(Exception):  # pydantic.ValidationError
            YOLOConfig(train="train.txt", val="val.txt", nc=3)  # missing 'names'

    def test_from_yaml(self, tmp_path):
        """Should load config from a valid YAML file."""
        config_data = {
            "train": "images/train.txt",
            "val": "images/val.txt",
            "nc": 2,
            "names": ["cat", "dog"],
        }
        yaml_path = tmp_path / "config.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(config_data, f)

        config = YOLOConfig.from_yaml(yaml_path)

        assert config.train == "images/train.txt"
        assert config.val == "images/val.txt"
        assert config.nc == 2
        assert config.names == ["cat", "dog"]

    def test_from_yaml_missing_key(self, tmp_path):
        """A YAML file missing required keys should raise."""
        yaml_path = tmp_path / "bad.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump({"train": "t.txt"}, f)

        with pytest.raises(Exception):
            YOLOConfig.from_yaml(yaml_path)


# ---------------------------------------------------------------------------
# YOLOConverter.__init__ / _get_config
# ---------------------------------------------------------------------------
class TestYOLOConverterInit:
    """Tests for YOLOConverter initialization and _get_config."""

    def test_init_with_yaml_file(self, yolo_dataset):
        """Should initialize successfully with a valid YAML config file."""
        config_path, _ = yolo_dataset
        converter = YOLOConverter(
            input=config_path,
            root_label="specimen",
            ext="arb",
            output=config_path.parent / "output",
        )

        assert converter.input == config_path
        assert converter.root_label == "specimen"
        assert converter.ext == "arb"

    def test_init_with_directory(self, yolo_dataset):
        """Should find the single .yaml in a directory."""
        config_path, _ = yolo_dataset
        converter = YOLOConverter(
            input=config_path.parent,
            root_label="specimen",
            ext="arb",
            output=config_path.parent / "output",
        )

        # After init, input should be resolved to the yaml file
        assert converter.input == config_path

    def test_nonexistent_path_raises(self, tmp_path):
        """Should raise ValueError for a non-existent input path."""
        with pytest.raises(ValueError, match="neither a file nor a directory"):
            YOLOConverter(
                input=tmp_path / "nonexistent.yaml",
                root_label="specimen",
                ext="arb",
                output=tmp_path / "output",
            )

    def test_non_yaml_file_raises(self, tmp_path):
        """Should raise ValueError for a non-YAML file."""
        bad_file = tmp_path / "config.json"
        bad_file.write_text("{}")

        with pytest.raises(ValueError, match="Expected a YAML config file"):
            YOLOConverter(
                input=bad_file,
                root_label="specimen",
                ext="arb",
                output=tmp_path / "output",
            )

    def test_no_yaml_in_dir_raises(self, tmp_path):
        """Should raise ValueError if no .yaml files exist in directory."""
        empty_dir = tmp_path / "dataset"
        empty_dir.mkdir()

        with pytest.raises(ValueError, match="No YAML config files found"):
            YOLOConverter(
                input=empty_dir,
                root_label="specimen",
                ext="arb",
                output=tmp_path / "output",
            )

    def test_multiple_yaml_in_dir_raises(self, tmp_path):
        """Should raise ValueError if multiple .yaml files in directory."""
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        (dataset_dir / "a.yaml").write_text(
            yaml.dump({"train": "t.txt", "val": "v.txt", "nc": 1, "names": ["x"]})
        )
        (dataset_dir / "b.yaml").write_text(
            yaml.dump({"train": "t.txt", "val": "v.txt", "nc": 1, "names": ["x"]})
        )

        with pytest.raises(ValueError, match="Multiple YAML config files"):
            YOLOConverter(
                input=dataset_dir,
                root_label="specimen",
                ext="arb",
                output=tmp_path / "output",
            )


# ---------------------------------------------------------------------------
# YOLOConverter._convert_file
# ---------------------------------------------------------------------------
class TestYOLOConverterConvertFile:
    """Tests for _convert_file — parsing YOLO annotations."""

    def test_converts_annotations_to_collectra_format(self, yolo_dataset, tmp_path):
        """YOLO annotation lines should be parsed into ImageCrop entries."""
        config_path, dataset_dir = yolo_dataset
        output = tmp_path / "output"

        converter = YOLOConverter(
            input=config_path,
            root_label="specimen",
            ext="arb",
            output=output,
        )

        img_file = dataset_dir / "images" / "img_0.jpg"
        txt_file = dataset_dir / "labels" / "img_0.txt"

        # Patch CollectraFile.from_file to capture what's built
        mock_cf = MagicMock()
        mock_cf.data = {"specimen": {"type": "collectra.Image", "data": "img_0.jpg"}}
        with patch(
            "collectra.converters.yolo.CollectraFile.from_file", return_value=mock_cf
        ):
            converter._convert_file(img_file, txt_file, partition="train")

        mock_cf.save.assert_called_once()
        # The annotation file has "0 0.5 0.5 0.1 0.2" and "1 0.3 0.7 0.05 0.1"
        assert "cat" in mock_cf.data
        assert "dog" in mock_cf.data
        assert len(mock_cf.data["cat"]) == 1
        assert len(mock_cf.data["dog"]) == 1

        cat_crop = mock_cf.data["cat"][0]
        assert cat_crop["type"] == "collectra.ImageCrop"
        assert cat_crop["x_center"] == 0.5
        assert cat_crop["y_center"] == 0.5
        assert cat_crop["width_relative"] == 0.1
        assert cat_crop["height_relative"] == 0.2
        assert cat_crop["parents"] == "specimen"

    def test_multiple_crops_same_class(self, yolo_dataset, tmp_path):
        """Multiple annotations for the same class should create a list."""
        config_path, dataset_dir = yolo_dataset
        output = tmp_path / "output"

        # Write an annotation with two entries for class 0
        txt_file = dataset_dir / "labels" / "img_0.txt"
        txt_file.write_text("0 0.5 0.5 0.1 0.2\n0 0.3 0.4 0.15 0.25\n")

        converter = YOLOConverter(
            input=config_path,
            root_label="specimen",
            ext="arb",
            output=output,
        )

        img_file = dataset_dir / "images" / "img_0.jpg"

        mock_cf = MagicMock()
        mock_cf.data = {"specimen": {"type": "collectra.Image", "data": "img_0.jpg"}}
        with patch(
            "collectra.converters.yolo.CollectraFile.from_file", return_value=mock_cf
        ):
            converter._convert_file(img_file, txt_file, partition="train")

        assert "cat" in mock_cf.data
        assert len(mock_cf.data["cat"]) == 2


# ---------------------------------------------------------------------------
# YOLOConverter._process_files
# ---------------------------------------------------------------------------
class TestYOLOConverterProcessFiles:
    """Tests for _process_files — file list processing."""

    def test_processes_all_listed_files(self, yolo_dataset, tmp_path):
        """All files listed in the input text file should be processed."""
        config_path, dataset_dir = yolo_dataset
        output = tmp_path / "output"

        converter = YOLOConverter(
            input=config_path,
            root_label="specimen",
            ext="arb",
            output=output,
            force=True,
        )

        train_txt = dataset_dir / "train.txt"
        with patch.object(converter, "_convert_file") as mock_convert:
            converter._process_files(train_txt, partition="train")

            assert mock_convert.call_count == 3

    def test_raises_on_missing_image(self, yolo_dataset, tmp_path):
        """Should raise ValueError if a listed image file doesn't exist."""
        config_path, dataset_dir = yolo_dataset
        output = tmp_path / "output"

        # Add a non-existent file to train.txt
        train_txt = dataset_dir / "train.txt"
        with open(train_txt, "a") as f:
            f.write("images/nonexistent.jpg\n")

        converter = YOLOConverter(
            input=config_path,
            root_label="specimen",
            ext="arb",
            output=output,
        )

        with pytest.raises(ValueError, match="does not exist"):
            converter._process_files(train_txt, partition="train")

    def test_cleans_up_output_on_error(self, yolo_dataset, tmp_path):
        """On conversion error, the output directory should be removed."""
        config_path, dataset_dir = yolo_dataset
        output = tmp_path / "output"

        converter = YOLOConverter(
            input=config_path,
            root_label="specimen",
            ext="arb",
            output=output,
            force=True,
        )

        train_txt = dataset_dir / "train.txt"
        with patch.object(converter, "_convert_file", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError, match="Failed to process files"):
                converter._process_files(train_txt, partition="train")

        assert not output.exists()


# ---------------------------------------------------------------------------
# YOLOConverter.convert
# ---------------------------------------------------------------------------
class TestYOLOConverterConvert:
    """Tests for the top-level convert() method."""

    def test_processes_train_and_val(self, yolo_dataset, tmp_path):
        """convert() should call _process_files for both train and val sets."""
        config_path, dataset_dir = yolo_dataset
        output = tmp_path / "output"

        converter = YOLOConverter(
            input=config_path,
            root_label="specimen",
            ext="arb",
            output=output,
            force=True,
        )

        with patch.object(converter, "_process_files") as mock_process:
            converter.convert()

            assert mock_process.call_count == 2
            # First call: train
            first_call = mock_process.call_args_list[0]
            assert "train" in str(first_call)
            # Second call: val
            second_call = mock_process.call_args_list[1]
            assert "val" in str(second_call)


# ---------------------------------------------------------------------------
# convert_files (module-level function)
# ---------------------------------------------------------------------------
class TestConvertFiles:
    """Tests for the convert_files() entry point."""

    def test_unknown_converter_raises(self, tmp_path):
        """Should raise ValueError for an unregistered converter name."""
        with pytest.raises(ValueError, match="not found"):
            convert_files(
                input=tmp_path,
                root_label="specimen",
                ext="arb",
                converter_name="nonexistent",
                output=tmp_path / "output",
            )

    def test_yolo_converter_invoked(self, tmp_path):
        """Should instantiate YOLOConverter and call convert() for 'yolo'."""
        with patch(
            "collectra.converters.converters",
            {"yolo": MagicMock()},
        ) as mock_converters:
            mock_cls = mock_converters["yolo"]
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance

            convert_files(
                input=tmp_path / "config.yaml",
                root_label="specimen",
                ext="arb",
                converter_name="yolo",
                output=tmp_path / "output",
                force=True,
            )

            mock_cls.assert_called_once_with(
                tmp_path / "config.yaml",
                "specimen",
                "arb",
                tmp_path / "output",
                force=True,
            )
            mock_instance.convert.assert_called_once()

    def test_error_message_lists_available_converters(self, tmp_path):
        """Error message for unknown converter should list available ones."""
        with pytest.raises(ValueError, match="yolo"):
            convert_files(
                input=tmp_path,
                root_label="specimen",
                ext="arb",
                converter_name="bad",
                output=tmp_path / "output",
            )
