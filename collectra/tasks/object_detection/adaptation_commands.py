"""CLI commands for adapting detector checkpoints to aligned RGB views."""

from pathlib import Path

from collectra.cli import command


def _output_path(input_model, output_model, num_views, suffix, overwrite):
    input_model = Path(input_model)
    if not input_model.is_file():
        raise ValueError(f"Input checkpoint does not exist: {input_model}")
    if num_views < 1:
        raise ValueError("num_views must be >= 1")
    output_model = (
        Path(output_model)
        if output_model
        else input_model.with_name(
            f"{input_model.stem}-{num_views}views-{suffix}{input_model.suffix or '.pth'}"
        )
    )
    if input_model.resolve() == output_model.resolve():
        raise ValueError("Output must differ from the input checkpoint")
    if output_model.exists() and not overwrite:
        raise ValueError("Output already exists; use --overwrite")
    return output_model


def _check_fusion(num_views, reference_view, hidden_dim):
    if not 0 <= reference_view < num_views:
        raise ValueError("reference_view must be between 0 and num_views - 1")
    if hidden_dim < 1:
        raise ValueError("hidden_dim must be >= 1")


class RFDETRAdaptationCommands:
    @command
    def adapt_input_channels(
        self,
        input_model: Path,
        num_views: int,
        output: Path | None = None,
        variant: str | None = None,
        overwrite: bool = False,
        validate: bool = True,
    ) -> Path:
        """Expand RGB input filters across aligned views and save a checkpoint."""
        destination = _output_path(
            input_model, output, num_views, f"{3 * num_views}ch", overwrite
        )
        from .rfdetr_input_adaptation import (
            expand_rfdetr_input_channels,
            validate_saved_model,
        )

        expand_rfdetr_input_channels(input_model, destination, num_views, variant)
        if validate:
            validate_saved_model(destination, 3 * num_views)
        print(f"Saved: {destination}")
        return destination

    @command
    def adapt_feature_fusion(
        self,
        input_model: Path,
        num_views: int,
        output: Path | None = None,
        variant: str | None = None,
        reference_view: int = 0,
        hidden_dim: int = 64,
        overwrite: bool = False,
        validate: bool = True,
    ) -> Path:
        """Add shared RGB encoding and attention across aligned views."""
        destination = _output_path(input_model, output, num_views, "fusion", overwrite)
        _check_fusion(num_views, reference_view, hidden_dim)
        from .rfdetr_feature_fusion import (
            expand_rfdetr_feature_fusion,
            validate_saved_model,
        )

        expand_rfdetr_feature_fusion(
            input_model, destination, num_views, variant, reference_view, hidden_dim
        )
        if validate:
            validate_saved_model(destination, num_views)
        print(f"Saved: {destination}")
        return destination


class YOLOAdaptationCommands:
    @command
    def adapt_feature_fusion(
        self,
        input_model: Path,
        num_views: int,
        output: Path | None = None,
        reference_view: int = 0,
        hidden_dim: int = 64,
        overwrite: bool = False,
        validate: bool = True,
    ) -> Path:
        """Add a shared RGB backbone and attention across aligned views."""
        destination = _output_path(input_model, output, num_views, "fusion", overwrite)
        if Path(input_model).suffix.lower() != ".pt":
            raise ValueError("Input must be an RGB YOLO .pt checkpoint")
        _check_fusion(num_views, reference_view, hidden_dim)
        from .yolo_feature_fusion import (
            expand_yolo_feature_fusion,
            validate_saved_model,
        )

        expand_yolo_feature_fusion(
            input_model, destination, num_views, reference_view, hidden_dim
        )
        if validate:
            validate_saved_model(destination, num_views)
        print(f"Saved: {destination}")
        return destination
