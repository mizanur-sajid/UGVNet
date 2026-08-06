import pytest
import torch

from ugvnet.hybrid import ugvnet_hybrid


def test_hybrid_classifier_shape() -> None:
    model = ugvnet_hybrid(
        num_classes=7,
        pretrained=False,
        fusion_channels=64,
        attention_heads=4,
        fusion_depth=1,
    ).eval()
    with torch.inference_mode():
        output = model(torch.randn(1, 3, 64, 64))
    assert output.shape == (1, 7)


def test_fusion_weights_form_probability_distribution() -> None:
    model = ugvnet_hybrid(
        num_classes=3,
        pretrained=False,
        fusion_channels=64,
        attention_heads=4,
        fusion_depth=1,
    ).eval()
    with torch.inference_mode():
        features, weights = model.forward_features(
            torch.randn(1, 3, 64, 64),
            return_fusion_weights=True,
        )
    assert features.shape[1] == 64
    assert weights.shape[1] == 2
    assert torch.allclose(
        weights.sum(dim=1),
        torch.ones_like(weights[:, 0]),
        atol=1e-5,
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"fusion_channels": 0}, "fusion_channels must be positive"),
        ({"attention_heads": 0}, "attention_heads must be positive"),
        ({"dropout": 1.0}, "dropout must be in"),
    ],
)
def test_invalid_hybrid_configuration_is_rejected(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        ugvnet_hybrid(num_classes=2, pretrained=False, **kwargs)


def test_backbone_freezing() -> None:
    model = ugvnet_hybrid(
        num_classes=2,
        pretrained=False,
        fusion_channels=64,
        attention_heads=4,
        fusion_depth=1,
    )
    model.set_backbones_trainable(False)
    assert not any(parameter.requires_grad for parameter in model.backbone_parameters())
    assert all(parameter.requires_grad for parameter in model.new_parameters())
