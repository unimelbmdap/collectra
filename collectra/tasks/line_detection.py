"""Deterministic printed-text line detection."""

from __future__ import annotations

from typing import Any

import numpy as np

from collectra.tasks.base import Task
from collectra.types.images import Image, ImageCrop, Orientation

__all__ = ["LineDetectorRLSA"]


class LineDetectorRLSA(Task):
    """Split printed text into lines using smearing and projection profiles.

    The detector is deterministic and does not load a machine-learning model.
    All size parameters are relative to the input dimensions so that the same
    configuration works across scan resolutions.
    """

    def __init__(
        self,
        name: str,
        deskew: bool = True,
        max_skew_degrees: float = 5.0,
        adaptive_block_fraction: float = 0.03,
        adaptive_constant: float = 15.0,
        rule_length_fraction: float = 0.35,
        smear_width_fraction: float = 0.015,
        smear_height_fraction: float = 0.003,
        smooth_height_fraction: float = 0.002,
        min_ink_fraction: float = 0.002,
        min_component_area_fraction: float = 0.000002,
        min_line_height_fraction: float = 0.003,
        max_line_gap_fraction: float = 0.004,
        padding_fraction: float = 0.003,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, **kwargs)
        self.deskew = deskew
        self.max_skew_degrees = max_skew_degrees
        self.adaptive_block_fraction = adaptive_block_fraction
        self.adaptive_constant = adaptive_constant
        self.rule_length_fraction = rule_length_fraction
        self.smear_width_fraction = smear_width_fraction
        self.smear_height_fraction = smear_height_fraction
        self.smooth_height_fraction = smooth_height_fraction
        self.min_ink_fraction = min_ink_fraction
        self.min_component_area_fraction = min_component_area_fraction
        self.min_line_height_fraction = min_line_height_fraction
        self.max_line_gap_fraction = max_line_gap_fraction
        self.padding_fraction = padding_fraction
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        fractions = (
            "adaptive_block_fraction",
            "rule_length_fraction",
            "smear_width_fraction",
            "smear_height_fraction",
            "smooth_height_fraction",
            "min_ink_fraction",
            "min_component_area_fraction",
            "min_line_height_fraction",
            "max_line_gap_fraction",
            "padding_fraction",
        )
        for name in fractions:
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1, got {value}")
        if not 0 <= self.max_skew_degrees <= 45:
            raise ValueError("max_skew_degrees must be between 0 and 45")

    def input_type(self) -> type:
        return Image

    def output_type(self) -> type:
        return ImageCrop

    @staticmethod
    def _odd_size(value: float, minimum: int = 3) -> int:
        size = max(minimum, int(round(value)))
        return size if size % 2 else size + 1

    def _binarize(self, image: np.ndarray) -> np.ndarray:
        import cv2

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        block_size = self._odd_size(
            min(gray.shape) * self.adaptive_block_fraction, minimum=15
        )
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            self.adaptive_constant,
        )
        return self._remove_noise(binary)

    def _remove_noise(self, binary: np.ndarray) -> np.ndarray:
        import cv2

        height, width = binary.shape
        minimum_area = max(2, round(height * width * self.min_component_area_fraction))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        cleaned = np.zeros_like(binary)
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] >= minimum_area:
                cleaned[labels == label] = 255
        return cleaned

    def _remove_horizontal_rules(self, binary: np.ndarray) -> np.ndarray:
        import cv2

        width = binary.shape[1]
        kernel_width = max(3, round(width * self.rule_length_fraction))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
        rules = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        return cv2.subtract(binary, rules)

    def _estimate_skew(self, binary: np.ndarray) -> float:
        import cv2

        if not self.deskew or not np.any(binary):
            return 0.0
        height, width = binary.shape
        lines = cv2.HoughLinesP(
            binary,
            1,
            np.pi / 720,
            threshold=max(20, width // 20),
            minLineLength=max(20, width // 12),
            maxLineGap=max(5, width // 100),
        )
        if lines is None:
            return 0.0
        angles = []
        for x1, y1, x2, y2 in lines[:, 0]:
            angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if abs(angle) <= self.max_skew_degrees:
                angles.append(angle)
        return float(np.median(angles)) if angles else 0.0

    @staticmethod
    def _rotate(binary: np.ndarray, angle: float) -> np.ndarray:
        import cv2

        if abs(angle) < 0.01:
            return binary
        height, width = binary.shape
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        return cv2.warpAffine(
            binary,
            matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    def _line_bands(self, binary: np.ndarray) -> list[tuple[int, int]]:
        import cv2

        height, width = binary.shape
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                max(1, round(width * self.smear_width_fraction)),
                max(1, round(height * self.smear_height_fraction)),
            ),
        )
        smeared = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        profile = np.count_nonzero(smeared, axis=1).astype(float)
        smooth_size = self._odd_size(height * self.smooth_height_fraction)
        profile = np.convolve(profile, np.ones(smooth_size) / smooth_size, mode="same")
        occupied = profile >= max(1, width * self.min_ink_fraction)

        ranges: list[tuple[int, int]] = []
        starts = np.flatnonzero(occupied & ~np.r_[False, occupied[:-1]])
        ends = np.flatnonzero(occupied & ~np.r_[occupied[1:], False]) + 1
        min_height = max(1, round(height * self.min_line_height_fraction))
        for top, bottom in zip(starts, ends):
            if bottom - top >= min_height:
                ranges.append((int(top), int(bottom)))

        max_gap = round(height * self.max_line_gap_fraction)
        merged: list[tuple[int, int]] = []
        for top, bottom in ranges:
            if merged and top - merged[-1][1] <= max_gap:
                merged[-1] = (merged[-1][0], bottom)
            else:
                merged.append((top, bottom))
        return merged

    def _crop_bounds(
        self, bands: list[tuple[int, int]], width: int, height: int, skew: float
    ) -> list[tuple[int, int, int, int]]:
        if not bands:
            return []
        skew_padding = abs(np.tan(np.radians(skew))) * width / 2
        padding = round(height * self.padding_fraction + skew_padding)
        boundaries = [0]
        boundaries.extend(
            (left[1] + right[0]) // 2 for left, right in zip(bands, bands[1:])
        )
        boundaries.append(height)
        return [
            (
                0,
                max(0, min(top - padding, band_top - padding)),
                width,
                min(height, max(bottom + padding, band_bottom + padding)),
            )
            for (band_top, band_bottom), top, bottom in zip(
                bands, boundaries, boundaries[1:]
            )
        ]

    def _make_crop(
        self,
        image: Image,
        bounds: tuple[int, int, int, int],
        display_width: int,
        display_height: int,
    ) -> ImageCrop:
        left, top, right, bottom = bounds
        local_x = (left + right) / (2 * display_width)
        local_y = (top + bottom) / (2 * display_height)
        local_width = (right - left) / display_width
        local_height = (bottom - top) / display_height

        display_left = local_x - local_width / 2
        display_top = local_y - local_height / 2
        display_right = local_x + local_width / 2
        display_bottom = local_y + local_height / 2
        display_corners = (
            (display_left, display_top),
            (display_right, display_top),
            (display_right, display_bottom),
            (display_left, display_bottom),
        )

        def to_source(x: float, y: float) -> tuple[float, float]:
            match image.orientation:
                case Orientation.WEST:
                    return y, 1 - x
                case Orientation.SOUTH:
                    return 1 - x, 1 - y
                case Orientation.EAST:
                    return 1 - y, x
                case _:
                    return x, y

        source_corners = [to_source(x, y) for x, y in display_corners]
        source_left = min(point[0] for point in source_corners)
        source_top = min(point[1] for point in source_corners)
        source_right = max(point[0] for point in source_corners)
        source_bottom = max(point[1] for point in source_corners)
        source_x = (source_left + source_right) / 2
        source_y = (source_top + source_bottom) / 2
        source_width = source_right - source_left
        source_height = source_bottom - source_top

        if isinstance(image, ImageCrop):
            return ImageCrop(
                name=self.get_output_name(),
                data=image.data,
                x_center=image.x_center + (source_x - 0.5) * image.width_relative,
                y_center=image.y_center + (source_y - 0.5) * image.height_relative,
                width_relative=source_width * image.width_relative,
                height_relative=source_height * image.height_relative,
                orientation=image.orientation,
            )
        return image.make_crop(
            source_x,
            source_y,
            source_width,
            source_height,
            name=self.get_output_name(),
            orientation=image.orientation,
        )

    def run(self, *images: Image) -> list[ImageCrop]:
        """Return one axis-aligned crop for each detected printed-text line."""
        results: list[ImageCrop] = []
        for image in images:
            if not isinstance(image, Image):
                raise TypeError(f"image must be Image, got {type(image)}")
            array = np.asarray(image.pil().convert("RGB"))
            binary = self._binarize(array)
            skew = self._estimate_skew(binary)
            deskewed = self._rotate(binary, -skew)
            bands = self._line_bands(self._remove_horizontal_rules(deskewed))
            bounds = self._crop_bounds(bands, array.shape[1], array.shape[0], skew)
            if not bounds:
                bounds = [(0, 0, array.shape[1], array.shape[0])]
            results.extend(
                self._make_crop(image, bound, array.shape[1], array.shape[0])
                for bound in bounds
            )
        return results
