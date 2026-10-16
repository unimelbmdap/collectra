"""Adapt Meta's installed image-training template for mask-supervised COCO data."""

from pathlib import Path
from types import SimpleNamespace

import yaml


def training_config(dataset, output, checkpoint, *, epochs, batch, num_gpus):
    from .sam3 import _sam3

    package = Path(_sam3().__file__).parent
    template = (
        package / "train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml"
    )
    if not template.is_file():
        raise FileNotFoundError(
            f"Install SAM3's training sources; template missing: {template}"
        )
    config = yaml.safe_load(template.read_text())
    config["paths"].update(
        roboflow_vl_100_root=str(dataset),
        experiment_log_dir=str(output / "training"),
        bpe_path=str(package / "assets/bpe_simple_vocab_16e6.txt.gz"),
    )
    settings = config["roboflow_train"]
    settings.update(supercategory="collectra", num_images=None)
    losses = settings["loss"]["loss_fns_find"]
    losses.append(
        {
            "_target_": "sam3.train.loss.loss_fns.Masks",
            "focal_alpha": 0.25,
            "focal_gamma": 2.0,
            "weight_dict": {"loss_mask": 200.0, "loss_dice": 10.0},
            "compute_aux": False,
        }
    )
    # Validation masks must be decoded too; Meta's detection template only
    # decodes them in the training transform sequence.
    settings["val_transforms"][0]["transforms"].insert(
        0,
        {
            "_target_": "sam3.train.transforms.segmentation.DecodeRle",
        },
    )
    config["scratch"].update(
        enable_segmentation=True,
        train_batch_size=batch,
        num_train_workers=0,
        num_val_workers=0,
    )
    trainer = config["trainer"]
    trainer.update(max_epochs=epochs, skip_saving_ckpts=False, val_epoch_freq=1)
    trainer["model"] = {
        "_target_": "collectra.tasks.image_segmentation.sam3_training.build_training_model",
        "checkpoint": str(checkpoint),
        "bpe_path": config["paths"]["bpe_path"],
    }
    for split, folder in (("train", "train"), ("val", "valid")):
        data = trainer["data"][split]
        data["dataset"].update(
            img_folder=str(dataset / folder),
            ann_file=str(dataset / folder / "_annotations.coco.json"),
            load_segmentation=True,
        )
        data["drop_last"] = False
    import json

    valid = json.loads((dataset / "valid/_annotations.coco.json").read_text())
    if valid["images"]:
        evaluator = trainer["meters"]["val"]["roboflow100"]["detection"][
            "pred_file_evaluators"
        ][0]
        evaluator["gt_path"] = str(dataset / "valid/_annotations.coco.json")
    else:
        # Training without held-out images must not silently evaluate on training data.
        trainer["data"].pop("val", None)
        trainer["meters"] = {}
        trainer["mode"] = "train_only"
    config["launcher"].update(gpus_per_node=num_gpus, num_nodes=1)
    config["submitit"].update(use_cluster=False, job_array=None)
    return config


def build_training_model(checkpoint, bpe_path):
    import torch
    from sam3.model_builder import build_sam3_image_model

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = state.get("model", state)
    if any(key.startswith("detector.") for key in state):
        state = {
            key.removeprefix("detector."): value
            for key, value in state.items()
            if key.startswith("detector.")
        }
    model = build_sam3_image_model(
        bpe_path=bpe_path,
        load_from_HF=False,
        enable_segmentation=True,
        device="cpu",
        eval_mode=False,
    )
    model.load_state_dict(state, strict=True)
    return model


def launch(config_path):
    from hydra import initialize_config_dir
    from sam3.train.train import main, register_omegaconf_resolvers

    config_path = Path(config_path).resolve()
    register_omegaconf_resolvers()
    with initialize_config_dir(str(config_path.parent), version_base="1.2"):
        main(
            SimpleNamespace(
                config=config_path.name, use_cluster=False, num_gpus=None, num_nodes=1
            )
        )
