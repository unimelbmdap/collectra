"""Turning annotated images into the COCO detection dataset a trainer reads."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from collectra.types.images import (
    Image,
    ImageCrop,
    bake_orientation,
    convert_image_to_rgb,
    image_channel_count,
    image_size,
)

from ...logger import get_logger
from ..machine_learning.training import print_distribution_table

logger = get_logger(__name__)


class CocoDetectionData:
    """Groups annotated crops by source image and writes train/valid COCO folders."""

    def _check_distribution(
        self,
        classes: list[str],
        train_samples: list[dict],
        val_samples: list[dict],
    ) -> None:
        train_classes = {name: 0 for name in classes}
        val_classes = {name: 0 for name in classes}
        label_to_name = {index: name for index, name in enumerate(classes)}

        for sample in train_samples:
            for label in sample["labels"]:
                class_name = label_to_name.get(label, "")
                if class_name:
                    train_classes[class_name] += 1

        for sample in val_samples:
            for label in sample["labels"]:
                class_name = label_to_name.get(label, "")
                if class_name:
                    val_classes[class_name] += 1

        def image_counts(samples: list[dict]) -> tuple[int, int]:
            labelled = sum(1 for sample in samples if sample["labels"])
            return labelled, len(samples) - labelled

        print_distribution_table(
            "Class Distribution",
            classes,
            train_classes,
            val_classes,
            train_images=image_counts(train_samples),
            validation_images=image_counts(val_samples),
        )

    def _prepare_assets(
        self,
        classes: list[str],
        validation_flag: str,
        exclude_flag: str,
        *images: ImageCrop,
    ) -> tuple[list[dict], list[dict]]:
        class_to_idx = {name: idx for idx, name in enumerate(classes)}
        grouped: dict[Path, dict] = {}

        for img in images:
            if exclude_flag and img.partition == exclude_flag:
                continue

            base_img: Image | ImageCrop = img
            if isinstance(img, ImageCrop):
                if img.source_parent:
                    base_img = img.source_parent
                    if isinstance(img.source_parent, ImageCrop):
                        img.set_rel_to_src_parent()

            image_path = base_img.get_path()
            if image_path not in grouped:
                grouped[image_path] = {
                    "image_path": image_path,
                    "partition": img.partition,
                    "boxes": [],
                    "labels": [],
                }

            if not isinstance(img, ImageCrop):
                continue
            if img.name not in class_to_idx:
                continue

            width, height = base_img.width, base_img.height
            box_w = float(img.width_relative) * width
            box_h = float(img.height_relative) * height
            cx = float(img.x_center) * width
            cy = float(img.y_center) * height

            x1 = max(0.0, cx - box_w / 2.0)
            y1 = max(0.0, cy - box_h / 2.0)
            x2 = min(float(width), cx + box_w / 2.0)
            y2 = min(float(height), cy + box_h / 2.0)
            if x2 <= x1 or y2 <= y1:
                continue

            grouped[image_path]["boxes"].append([x1, y1, x2, y2])
            grouped[image_path]["labels"].append(class_to_idx[img.name])

        train_samples: list[dict] = []
        val_samples: list[dict] = []
        for sample in grouped.values():
            if validation_flag and sample["partition"] == validation_flag:
                val_samples.append(sample)
            elif not exclude_flag or sample["partition"] != exclude_flag:
                train_samples.append(sample)
        return train_samples, val_samples

    def _write_coco_dataset(
        self,
        train_samples: list[dict],
        val_samples: list[dict],
        classes: list[str],
        dataset_dir: Path,
        rgb: bool = False,
    ) -> Path:
        categories = [
            {"id": idx, "name": name, "supercategory": "object"}
            for idx, name in enumerate(classes)
        ]

        channel_counts: dict[str, int] = {}
        for split_name, samples in [("train", train_samples), ("valid", val_samples)]:
            split_dir = dataset_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

            images_list: list[dict] = []
            annotations_list: list[dict] = []
            annotation_id = 0

            for image_id, sample in enumerate(samples):
                src_path = Path(sample["image_path"])
                dest_name = f"{image_id}_{src_path.name}"
                dest_path = split_dir / dest_name
                shutil.copy2(src_path, dest_path)
                # RF-DETR reads raw pixels; labels are in displayed coordinates.
                bake_orientation(dest_path)
                if rgb:
                    convert_image_to_rgb(dest_path)

                w, h = image_size(dest_path)
                channel_counts[str(src_path)] = image_channel_count(dest_path)

                images_list.append(
                    {
                        "id": image_id,
                        "file_name": dest_name,
                        "width": w,
                        "height": h,
                    }
                )

                for box, label in zip(sample["boxes"], sample["labels"]):
                    x1, y1, x2, y2 = box
                    coco_bbox = [x1, y1, x2 - x1, y2 - y1]
                    area = (x2 - x1) * (y2 - y1)

                    annotations_list.append(
                        {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": int(label),
                            "bbox": coco_bbox,
                            "area": float(area),
                            "iscrowd": 0,
                        }
                    )
                    annotation_id += 1

            coco_json = {
                "images": images_list,
                "annotations": annotations_list,
                "categories": categories,
            }

            annotations_path = split_dir / "_annotations.coco.json"
            with open(annotations_path, "w") as f:
                json.dump(coco_json, f, indent=2)

        if len(set(channel_counts.values())) > 1:
            details = ", ".join(
                f"{name}: {count} channels" for name, count in channel_counts.items()
            )
            raise ValueError(
                f"Inconsistent input channel counts in prepared dataset: {details}"
            )
        if channel_counts:
            logger.info(
                "Dataset images have %d channels",
                next(iter(channel_counts.values())),
            )
        return dataset_dir
