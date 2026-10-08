"""Verify reference preservation, attention gradients and RF-DETR reconstruction."""

import copy
from pathlib import Path

import pytest
import torch

pytest.importorskip("rfdetr")
from rfdetr import RFDETRNano
from rfdetr.detr import RFDETR
from rfdetr.models.backbone.dinov2 import DinoV2
from rfdetr.training.module_model import RFDETRModelModule

from collectra.tasks.object_detection.rfdetr import load_rfdetr_model
from collectra.tasks.object_detection.rfdetr_channels import (
    multichannel_checkpoints,
    sync_channels,
)
from collectra.tasks.object_detection.rfdetr_feature_fusion import (
    ViewAttention,
    adapt_model,
    load_adapted_model,
    save_adapted_model,
)


@pytest.fixture
def detector():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    model = RFDETRNano(
        pretrain_weights=None, device="cpu", resolution=128, positional_encoding_size=8
    )
    model.model.model.eval()
    yield model
    torch.set_num_threads(previous)


def test_attention_preserves_reference_and_learns():
    fusion = ViewAttention(8, 3, 1, 4)
    features = torch.randn(2, 3, 8, 3, 4, requires_grad=True)
    torch.testing.assert_close(fusion(features), features[:, 1], rtol=0, atol=0)
    fusion(features).square().mean().backward()
    assert fusion.gate.grad.abs() > 0
    fusion.zero_grad()
    features.grad = None
    with torch.no_grad():
        fusion.gate.fill_(0.5)
    fusion(features).square().mean().backward()
    assert all(features.grad[:, view].abs().sum() > 0 for view in range(3))
    assert fusion.score[-1].weight.grad.abs().sum() > 0
    # Attention weights are shared over locations but not forced equal across views.
    with torch.no_grad():
        fusion.score[-1].weight.add_(fusion.score[-1].weight.grad)
    fusion.zero_grad()
    fusion(features).square().mean().backward()
    assert fusion.score[0].weight.grad.abs().sum() > 0


def test_real_encoder_reference_and_identical_views(detector):
    encoder = detector.model.model.backbone[0].encoder
    rgb = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        original = encoder(rgb)
    adapt_model(detector, 3, reference_view=1, hidden_dim=4)
    pixels = torch.cat([torch.randn_like(rgb), rgb, torch.randn_like(rgb)], dim=1)
    with torch.no_grad():
        fused = encoder(pixels)
    for expected, actual in zip(original, fused):
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    with torch.no_grad():
        for fusion in encoder.view_fusion:
            fusion.gate.fill_(0.8)
        repeated = encoder(rgb.repeat(1, 3, 1, 1))
    for expected, actual in zip(original, repeated):
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    assert encoder.encoder.embeddings.patch_embeddings.projection.in_channels == 3
    assert sync_channels(detector) == 9
    with pytest.raises(ValueError, match="Expected BCHW"):
        encoder(rgb)


def test_checkpoint_roundtrip_and_collectra_dispatch(detector, tmp_path):
    adapt_model(detector, 2, reference_view=1, hidden_dim=4)
    with torch.no_grad():
        detector.model.model.backbone[0].encoder.view_fusion[0].gate.fill_(0.4)
    detector.model.class_names = ["a", "b"]
    path = tmp_path / "fusion.pth"
    save_adapted_model(detector, path)
    reloaded = load_rfdetr_model(path)
    reloaded.model.model.eval()
    assert reloaded.model_config.num_channels == 6
    assert reloaded.model.class_names == ["a", "b"]
    assert len(reloaded.means) == 6
    for name, value in detector.model.model.state_dict().items():
        torch.testing.assert_close(reloaded.model.model.state_dict()[name], value)
    pixels = torch.randn(1, 6, 64, 64)
    with torch.no_grad():
        expected = detector.model.model.backbone[0].encoder(pixels)
        actual = reloaded.model.model.backbone[0].encoder(pixels)
    for a, b in zip(expected, actual):
        torch.testing.assert_close(a, b)
    payload = torch.load(path, weights_only=True)
    del payload["model"]["backbone.0.encoder.view_fusion.0.gate"]
    broken = tmp_path / "broken.pth"
    torch.save(payload, broken)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_adapted_model(broken)


def test_training_rebuild_keeps_attention_and_rgb_projection(
    detector, tmp_path, monkeypatch
):
    adapt_model(detector, 2, hidden_dim=4)
    encoder = detector.model.model.backbone[0].encoder
    with torch.no_grad():
        encoder.view_fusion[0].gate.fill_(0.7)
    original_path = detector.model_config.pretrain_weights
    seen = {}

    def upstream_train(self, **kwargs):
        assert Path(self.model_config.pretrain_weights).is_file()
        assert kwargs["augmentation_backend"] == "torchvision"
        config = self.get_train_config(
            dataset_dir=str(tmp_path), output_dir=str(tmp_path)
        )
        module = RFDETRModelModule(self.model_config, config)
        rebuilt = module.model.backbone[0].encoder
        assert rebuilt.encoder.embeddings.patch_embeddings.projection.in_channels == 3
        torch.testing.assert_close(
            rebuilt.view_fusion[0].gate, encoder.view_fusion[0].gate
        )
        args = type(
            "Args",
            (),
            dict(
                lr=1e-4,
                lr_encoder=1.5e-4,
                lr_vit_layer_decay=0.8,
                lr_component_decay=0.7,
                out_feature_indexes=self.model_config.out_feature_indexes,
                weight_decay=1e-4,
            ),
        )()
        groups = module.model.backbone[0].get_named_param_lr_pairs(args)
        assert groups["backbone.0.encoder.view_fusion.0.gate"]["lr"] == 1e-4
        # Upstream checkpoints may omit our custom metadata but keep the buffer.
        module_path = tmp_path / "trained.pth"
        proxy = copy.copy(self)
        proxy.model = copy.copy(self.model)
        proxy.model.model = module.model
        save_adapted_model(proxy, module_path)
        payload = torch.load(module_path, weights_only=True)
        payload.pop("feature_fusion")
        payload["args"]["notes"] = config.notes
        torch.save(payload, module_path)
        from rfdetr.utilities.state_dict import strip_checkpoint

        strip_checkpoint(module_path)
        assert "model_config" not in torch.load(module_path, weights_only=True)
        loaded = load_adapted_model(module_path)
        assert loaded.model_config.num_channels == 6
        seen["snapshot"] = self.model_config.pretrain_weights

    monkeypatch.setattr(RFDETR, "train", upstream_train)
    # Same nesting used by Collectra's training orchestration.
    with multichannel_checkpoints():
        detector.train(dataset_dir=str(tmp_path))
    assert seen and not Path(seen["snapshot"]).exists()
    assert detector.model_config.pretrain_weights == original_path
    with pytest.raises(ValueError, match="single-device"):
        detector.train(devices=2)


def test_script_adapts_validates_and_preserves_detection_output(detector, tmp_path):
    import runpy

    adapter = runpy.run_path("adapt-rfdetr-feature-fusion.py")
    rgb_path = tmp_path / "rgb.pth"
    config = detector.model_config.model_dump(mode="json")
    torch.save(
        {
            "model": detector.model.model.state_dict(),
            "model_config": config,
            "model_name": "RFDETRNano",
            "args": dict(config, class_names=[]),
        },
        rgb_path,
    )
    output = tmp_path / "fusion.pth"
    fused = adapter["expand_rfdetr_feature_fusion"](
        rgb_path, output, 2, reference_view=1, hidden_dim=4
    )
    adapter["validate_saved_model"](output, 2)
    fused.model.model.eval()
    rgb = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        expected = detector.model.model(rgb)
        actual = fused.model.model(torch.cat([torch.randn_like(rgb), rgb], dim=1))
    for key in ("pred_logits", "pred_boxes"):
        torch.testing.assert_close(actual[key], expected[key], atol=1e-4, rtol=1e-4)
