import pytest
import torch

from ugvnet import create_ugvnet, ugvnet_tiny


def test_classifier_output_shape() -> None:
    model = ugvnet_tiny(num_classes=7).eval()
    with torch.inference_mode():
        output = model(torch.randn(2, 3, 64, 64))
    assert output.shape == (2, 7)


def test_resolution_flexible_and_intermediate_shapes() -> None:
    model = ugvnet_tiny(num_classes=3).eval()
    with torch.inference_mode():
        features = model.forward_intermediates(torch.randn(1, 3, 96, 128))
    assert [feature.shape[1] for feature in features] == [48, 96, 192, 320]
    assert [feature.shape[-2:] for feature in features] == [
        (24, 32),
        (12, 16),
        (6, 8),
        (3, 4),
    ]


def test_backbone_mode_and_reset_classifier() -> None:
    model = create_ugvnet("tiny", num_classes=0).eval()
    with torch.inference_mode():
        embedding = model(torch.randn(1, 3, 64, 64))
    assert embedding.shape == (1, 320)
    model.reset_classifier(5)
    with torch.inference_mode():
        logits = model(torch.randn(1, 3, 64, 64))
    assert logits.shape == (1, 5)


def test_custom_input_channels() -> None:
    model = ugvnet_tiny(num_classes=2, in_channels=1).eval()
    with torch.inference_mode():
        output = model(torch.randn(1, 1, 64, 64))
    assert output.shape == (1, 2)


def test_reset_classifier_preserves_model_dtype() -> None:
    model = ugvnet_tiny(num_classes=0).double()
    model.reset_classifier(4)
    assert isinstance(model.head, torch.nn.Linear)
    assert model.head.weight.dtype == torch.float64


def test_invalid_input_channels_are_rejected() -> None:
    with pytest.raises(ValueError, match="in_channels must be positive"):
        ugvnet_tiny(num_classes=2, in_channels=0)


def test_unknown_variant_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown UGVNet variant"):
        create_ugvnet("large")
