from copy import deepcopy

import numpy as np
import pytest
import tifffile

from collectra import Image, ImageClassifierTorchvision, Link
from collectra.tasks.image_classifier.torchvision_fusion import (
    RGBViewFusionClassifier,
    SpatialViewAttention,
    restore_fusion,
)


@pytest.fixture(autouse=True)
def single_thread():
    import torch

    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def randomize_attention(model, seed=0):
    import torch

    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in model.attention.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator))


@pytest.mark.parametrize("architecture", ["resnet18", "convnext_tiny"])
def test_identical_views_reproduce_rgb_model_with_shared_encoder(architecture):
    import torch
    from torchvision import models

    original = models.get_model(architecture, weights=None).eval()
    reference = deepcopy(original)
    model = RGBViewFusionClassifier(original, 9).eval()
    randomize_attention(model)
    rgb = torch.randn(2, 3, 32, 32)
    observed = []
    handle = model.encoder.register_forward_pre_hook(
        lambda module, args: observed.append(args[0].shape)
    )
    with torch.no_grad():
        # Any convex combination of identical views is that view.
        torch.testing.assert_close(
            model(rgb.repeat(1, 3, 1, 1)), reference(rgb), rtol=1e-4, atol=1e-5
        )
    handle.remove()
    assert observed == [torch.Size([6, 3, 32, 32])]
    first = next(
        layer for layer in model.encoder.modules() if isinstance(layer, torch.nn.Conv2d)
    )
    assert first.in_channels == 3
    attention = sum(p.numel() for p in model.attention.parameters())
    assert sum(p.numel() for p in model.parameters()) - attention == sum(
        p.numel() for p in reference.parameters()
    )


def test_fusion_is_view_agnostic():
    from torchvision import models

    model = RGBViewFusionClassifier(models.resnet18(weights=None), 9)
    assert not hasattr(model, "reference_view")
    assert not hasattr(model, "view_bias")
    assert not hasattr(model, "gate")
    assert model.fusion_spec == {"version": 2, "num_views": 3, "hidden_dim": 128}


def test_initial_fusion_averages_spatial_feature_maps_before_pooling():
    import torch
    from torchvision import models

    original = models.resnet18(weights=None).eval()
    reference = deepcopy(original)
    model = RGBViewFusionClassifier(original, 9).eval()
    pixels = torch.randn(2, 9, 32, 32)
    maps = []
    handle = reference.avgpool.register_forward_pre_hook(
        lambda module, args: maps.append(args[0])
    )
    with torch.no_grad():
        for view in range(3):
            reference(pixels[:, 3 * view : 3 * view + 3])
        handle.remove()
        mean_map = torch.stack(maps).mean(0)
        expected = reference.fc(torch.flatten(reference.avgpool(mean_map), 1))
        torch.testing.assert_close(model(pixels), expected, rtol=1e-4, atol=1e-5)


def test_view_order_does_not_change_prediction():
    import torch
    from torchvision import models

    model = RGBViewFusionClassifier(models.resnet18(weights=None), 9).eval()
    randomize_attention(model, seed=1)
    pixels = torch.randn(2, 9, 32, 32)
    permuted = torch.cat([pixels[:, 6:9], pixels[:, 0:3], pixels[:, 3:6]], dim=1)
    with torch.no_grad():
        torch.testing.assert_close(model(permuted), model(pixels), rtol=1e-4, atol=1e-5)


def test_attention_weights_are_per_position_softmax_over_views():
    import torch

    attention = SpatialViewAttention(4, hidden_dim=8)
    features = torch.randn(2, 3, 4, 5, 6)
    weights = attention.weights(features)
    assert weights.shape == (2, 3, 5, 6)
    torch.testing.assert_close(weights, torch.full_like(weights, 1 / 3))
    torch.testing.assert_close(attention(features), features.mean(1))
    with torch.no_grad():
        for parameter in attention.parameters():
            parameter.copy_(torch.randn(parameter.shape))
    weights = attention.weights(features)
    torch.testing.assert_close(weights.sum(1), torch.ones(2, 5, 6))
    # Weights vary across positions, so each location chooses its own views.
    assert weights.flatten(2).std(-1).max() > 0
    expected = (features * weights[:, :, None]).sum(1)
    torch.testing.assert_close(attention(features), expected)


def test_attention_learns_from_every_view_and_freezing_preserves_backbone():
    import torch
    from torchvision import models

    task = ImageClassifierTorchvision(
        "classifier", model=models.resnet18(weights=None), fusion=True
    )
    task._categories = [str(index) for index in range(1000)]
    task._input_channels = 9
    task._prepare_training_model(["a", "b"], pretrained=False, freeze=True)
    model = task.model.eval()
    attention = {id(parameter) for parameter in model.attention.parameters()}
    assert attention and all(
        parameter.requires_grad for parameter in model.attention.parameters()
    )
    assert not any(
        parameter.requires_grad
        for parameter in model.encoder.parameters()
        if id(parameter) not in attention
    )
    assert all(parameter.requires_grad for parameter in model.head.parameters())
    randomize_attention(model, seed=2)
    pixels = torch.randn(2, 9, 32, 32, requires_grad=True)
    model(pixels).square().mean().backward()
    for view in range(3):
        assert pixels.grad[:, view * 3 : (view + 1) * 3].abs().sum() > 0
    assert model.attention.score[-1].weight.grad.abs().sum() > 0


def test_fusion_checkpoint_roundtrip_and_new_class_head(tmp_path):
    import torch
    from torchvision import models

    task = ImageClassifierTorchvision(
        "classifier", model=models.resnet18(weights=None), fusion=True
    )
    task._architecture = "resnet18"
    task._input_channels = 6
    task._categories = [str(index) for index in range(1000)]
    task._preprocessing = {
        "resize_size": [32],
        "crop_size": [32],
        "mean": [0.5] * 6,
        "std": [0.5] * 6,
        "interpolation": "bilinear",
    }
    task._prepare_training_model(["a", "b"], pretrained=False, freeze=False)
    task.model.eval()
    randomize_attention(task.model, seed=3)
    pixels = torch.randn(2, 6, 32, 32)
    expected = task.model(pixels).detach()
    checkpoint = tmp_path / "fusion.pt"
    torch.save(task._checkpoint(1, {}), checkpoint)
    loaded = ImageClassifierTorchvision("classifier", model=checkpoint)
    loaded._load()
    assert loaded.fusion is True
    assert loaded.model.fusion_spec == task.model.fusion_spec
    torch.testing.assert_close(loaded.model(pixels), expected)
    loaded._input_channels = 9
    with pytest.raises(ValueError, match="channel count differs"):
        loaded._prepare_training_model(["a", "b"], pretrained=False, freeze=False)
    loaded._input_channels = 6
    before = loaded.model.attention.score[-1].weight.detach().clone()
    loaded._prepare_training_model(["c", "b", "a"], pretrained=False, freeze=False)
    assert loaded.model.head.out_features == 3
    torch.testing.assert_close(loaded.model.attention.score[-1].weight, before)


@pytest.mark.parametrize(
    "spec,message",
    [
        (
            {"version": 1, "num_views": 2, "reference_view": 0, "hidden_dim": 8},
            "version",
        ),
        ("not a spec", "version"),
        ({"version": 2, "num_views": 3, "hidden_dim": 8}, "view count"),
    ],
)
def test_restore_rejects_old_or_mismatched_specs(spec, message):
    from torchvision import models

    with pytest.raises(ValueError, match=message):
        restore_fusion(models.resnet18(weights=None), 6, spec)


@pytest.mark.parametrize("channels", [1, 4, 5])
def test_fusion_rejects_incomplete_rgb_stacks(channels):
    from torchvision import models

    with pytest.raises(ValueError, match="multiple of 3"):
        RGBViewFusionClassifier(models.resnet18(weights=None), channels)


def test_fusion_rejects_bad_hidden_dim():
    from torchvision import models

    with pytest.raises(ValueError, match="hidden dimension"):
        RGBViewFusionClassifier(models.resnet18(weights=None), 6, hidden_dim=0)


def test_fusion_rejects_expanded_stem_and_wrong_inference_channels():
    import torch
    from torchvision import models

    expanded = models.resnet18(weights=None)
    expanded.conv1 = torch.nn.Conv2d(6, 64, 7)
    with pytest.raises(ValueError, match="RGB backbone"):
        RGBViewFusionClassifier(expanded, 6)
    model = RGBViewFusionClassifier(models.resnet18(weights=None), 6)
    with pytest.raises(ValueError, match="6 channels"):
        model(torch.randn(1, 3, 32, 32))


def test_fusion_requires_spatial_avgpool():
    import torch
    from torchvision import models

    with pytest.raises(ValueError, match="avgpool"):
        RGBViewFusionClassifier(models.vit_b_16(weights=None), 6)

    class UnusedPool(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 4, 1)
            self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
            self.fc = torch.nn.Linear(4, 2)

        def forward(self, pixels):
            return self.fc(self.conv1(pixels).mean((2, 3)))

    with pytest.raises(ValueError, match="avgpool"):
        RGBViewFusionClassifier(UnusedPool(), 6)


def test_channel_probe_preserves_training_modes():
    from torchvision import models

    model = models.resnet18(weights=None).train()
    model.layer1.eval()
    fused = RGBViewFusionClassifier(model, 6)
    assert fused.training and fused.encoder.layer2.training
    assert not fused.encoder.layer1.training


def test_yaml_fusion_training_tiff_inference_and_reload(tmp_path, monkeypatch):
    import torch
    from types import SimpleNamespace
    from torchvision import models, transforms
    from collectra import Pipeline

    class TinyClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 4, 1)
            self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
            self.fc = torch.nn.Linear(4, 1000)

        def forward(self, pixels):
            return self.fc(torch.flatten(self.avgpool(self.conv1(pixels)), 1))

    monkeypatch.setattr(models, "get_model", lambda *args, **kwargs: TinyClassifier())
    weights = SimpleNamespace(
        meta={"categories": [str(index) for index in range(1000)]},
        transforms=lambda: SimpleNamespace(
            crop_size=[8],
            resize_size=[8],
            mean=[0.5] * 3,
            std=[0.5] * 3,
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
    )
    monkeypatch.setattr(
        models, "get_model_weights", lambda name: SimpleNamespace(DEFAULT=weights)
    )
    pipeline = tmp_path / "pipeline.yaml"
    pipeline.write_text(
        'collectra_pipeline_metadata:\n  name: Fusion\n  ext: collectra\n  version: "1"\nclassifier:\n  type: collectra.ImageClassifierTorchvision\n  input: image\n  output: [a, b]\n  model: resnet18\n  fusion: true\n'
    )
    workflow = Pipeline.from_file(pipeline)
    workflow.connect()
    task = workflow.node_manager.resolve_node("classifier").get_task()
    assert task.fusion is True
    samples = []
    for index, (name, partition) in enumerate(
        [("a", "train"), ("b", "train"), ("a", "val"), ("b", "val")]
    ):
        image = tmp_path / f"{index}.tif"
        tifffile.imwrite(
            image,
            np.full((6, 8, 8), index + 1, dtype="uint8"),
            photometric="minisblack",
            metadata={"axes": "CYX"},
        )
        samples.append(Image(name=name, data=image, partition=partition))
    result = task._train(
        *samples,
        base_folder=tmp_path,
        log="run",
        validation="val",
        epochs=1,
        batch=2,
        workers=0,
        device="cpu",
        wandb=False,
        rotate90=True,
    )
    saved = ImageClassifierTorchvision(
        "classifier",
        model=result.save_dir / "weights/best.pt",
        output=["a", "b"],
        device="cpu",
    )
    prediction = saved.run(samples[0])
    assert isinstance(prediction, Link) and prediction.target is samples[0]
    assert saved.fusion is True and saved._input_channels == 6
    assert saved.model.encoder.conv1.in_channels == 3
    assert saved._preprocessing["mean"] == [0.5] * 6


def test_fusion_requires_final_linear_projection():
    import torch

    class ConvolutionalHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 4, 1)
            self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
            self.classifier = torch.nn.Sequential(torch.nn.Conv2d(4, 2, 1))

        def forward(self, pixels):
            return self.classifier(self.avgpool(self.conv1(pixels))).flatten(1)

    with pytest.raises(ValueError, match="final linear classification head"):
        RGBViewFusionClassifier(ConvolutionalHead(), 6)
