"""Recommended dual-backbone UGVNet architecture.

This module combines ImageNet-pretrained EfficientNetV2-S and ConvNeXt-Tiny
feature extractors. Their final spatial feature maps are aligned, projected,
adaptively fused, and refined with global self-attention.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    EfficientNet_V2_S_Weights,
    convnext_tiny,
    efficientnet_v2_s,
)

from .lightweight import DropPath, GlobalSelfAttention, LayerNorm2d


class FusionMLP(nn.Sequential):
    """Convolutional feed-forward network used after global fusion."""

    def __init__(self, channels: int, expansion: float, dropout: float) -> None:
        hidden_channels = int(channels * expansion)
        super().__init__(
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=hidden_channels,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
            nn.Dropout(dropout),
        )


class GlobalFusionBlock(nn.Module):
    """Resolution-flexible global-attention refinement block."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        drop_path: float = 0.0,
        layer_scale_init: float = 1e-5,
    ) -> None:
        super().__init__()
        self.position = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
        )
        self.attention_norm = LayerNorm2d(channels)
        self.attention = GlobalSelfAttention(
            channels,
            num_heads,
            attention_dropout=attention_dropout,
            projection_dropout=dropout,
        )
        self.mlp_norm = LayerNorm2d(channels)
        self.mlp = FusionMLP(channels, mlp_ratio, dropout)
        self.drop_path = DropPath(drop_path)
        self.attention_scale = nn.Parameter(
            torch.full((channels, 1, 1), layer_scale_init)
        )
        self.mlp_scale = nn.Parameter(torch.full((channels, 1, 1), layer_scale_init))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.position(x)
        x = x + self.drop_path(
            self.attention_scale * self.attention(self.attention_norm(x))
        )
        x = x + self.drop_path(self.mlp_scale * self.mlp(self.mlp_norm(x)))
        return x


class AdaptiveBackboneFusion(nn.Module):
    """Project and spatially gate two backbone feature maps."""

    def __init__(
        self,
        efficientnet_channels: int = 1280,
        convnext_channels: int = 768,
        fusion_channels: int = 384,
    ) -> None:
        super().__init__()
        self.efficientnet_projection = nn.Sequential(
            nn.Conv2d(efficientnet_channels, fusion_channels, kernel_size=1),
            nn.GroupNorm(1, fusion_channels),
            nn.GELU(),
        )
        self.convnext_projection = nn.Sequential(
            nn.Conv2d(convnext_channels, fusion_channels, kernel_size=1),
            nn.GroupNorm(1, fusion_channels),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(fusion_channels * 2, fusion_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(fusion_channels, 2, kernel_size=1),
        )

    def forward(
        self, efficientnet_features: Tensor, convnext_features: Tensor
    ) -> tuple[Tensor, Tensor]:
        efficientnet_features = self.efficientnet_projection(efficientnet_features)
        convnext_features = self.convnext_projection(convnext_features)
        if efficientnet_features.shape[-2:] != convnext_features.shape[-2:]:
            efficientnet_features = functional.interpolate(
                efficientnet_features,
                size=convnext_features.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        weights = self.gate(
            torch.cat((efficientnet_features, convnext_features), dim=1)
        ).softmax(dim=1)
        fused = (
            weights[:, 0:1] * efficientnet_features
            + weights[:, 1:2] * convnext_features
        )
        return fused, weights


class UGVNetHybrid(nn.Module):
    """UGVNet with EfficientNetV2-S and ConvNeXt-Tiny backbones.

    Args:
        num_classes: Number of output classes. Set to zero to return embeddings.
        pretrained: Load ImageNet-1K backbone weights.
        fusion_channels: Width of the shared feature space.
        attention_heads: Number of global-attention heads.
        fusion_depth: Number of global fusion-refinement blocks.
        dropout: Dropout used by fusion blocks and the classifier.
    """

    def __init__(
        self,
        num_classes: int,
        *,
        pretrained: bool = True,
        fusion_channels: int = 384,
        attention_heads: int = 8,
        fusion_depth: int = 2,
        dropout: float = 0.2,
        attention_dropout: float = 0.0,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()
        if num_classes < 0:
            raise ValueError("num_classes must be non-negative.")
        if fusion_channels < 1:
            raise ValueError("fusion_channels must be positive.")
        if attention_heads < 1:
            raise ValueError("attention_heads must be positive.")
        if fusion_channels % attention_heads != 0:
            raise ValueError("fusion_channels must be divisible by attention_heads.")
        if fusion_depth < 1:
            raise ValueError("fusion_depth must be at least one.")
        for name, value in (
            ("dropout", dropout),
            ("attention_dropout", attention_dropout),
            ("drop_path_rate", drop_path_rate),
        ):
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1).")

        efficientnet_weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        convnext_weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        efficientnet = efficientnet_v2_s(weights=efficientnet_weights)
        convnext = convnext_tiny(weights=convnext_weights)
        self.efficientnet_features = efficientnet.features
        self.convnext_features = convnext.features
        self.fusion = AdaptiveBackboneFusion(fusion_channels=fusion_channels)

        drop_paths = torch.linspace(0, drop_path_rate, fusion_depth).tolist()
        self.global_fusion = nn.Sequential(
            *[
                GlobalFusionBlock(
                    fusion_channels,
                    num_heads=attention_heads,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    drop_path=drop_paths[index],
                )
                for index in range(fusion_depth)
            ]
        )
        self.final_norm = LayerNorm2d(fusion_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.num_classes = num_classes
        self.embedding_dim = fusion_channels
        self.head: nn.Module = (
            nn.Linear(fusion_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self._initialize_new_layers()

    def _initialize_new_layers(self) -> None:
        modules = (self.fusion, self.global_fusion, self.head)
        for root in modules:
            for module in root.modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    nn.init.trunc_normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, (nn.GroupNorm, nn.LayerNorm)):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

    def extract_backbone_features(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return the two unfused stride-32 feature maps."""
        return self.efficientnet_features(x), self.convnext_features(x)

    def forward_features(
        self, x: Tensor, *, return_fusion_weights: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Return the fused spatial representation."""
        efficientnet_features, convnext_features = self.extract_backbone_features(x)
        fused, weights = self.fusion(efficientnet_features, convnext_features)
        fused = self.final_norm(self.global_fusion(fused))
        if return_fusion_weights:
            return fused, weights
        return fused

    def forward_head(self, features: Tensor) -> Tensor:
        embedding = self.pool(features).flatten(1)
        return self.head(self.dropout(embedding))

    def forward(self, x: Tensor) -> Tensor:
        features = cast(Tensor, self.forward_features(x))
        return self.forward_head(features)

    def set_backbones_trainable(self, trainable: bool) -> None:
        """Freeze or unfreeze both pretrained feature extractors."""
        for backbone in (self.efficientnet_features, self.convnext_features):
            for parameter in backbone.parameters():
                parameter.requires_grad = trainable

    def train(self, mode: bool = True) -> UGVNetHybrid:
        """Keep frozen backbones in evaluation mode during head warm-up."""
        super().train(mode)
        if mode:
            if not any(
                parameter.requires_grad
                for parameter in self.efficientnet_features.parameters()
            ):
                self.efficientnet_features.eval()
            if not any(
                parameter.requires_grad
                for parameter in self.convnext_features.parameters()
            ):
                self.convnext_features.eval()
        return self

    def backbone_parameters(self):
        """Iterate parameters that should use the lower fine-tuning LR."""
        yield from self.efficientnet_features.parameters()
        yield from self.convnext_features.parameters()

    def new_parameters(self):
        """Iterate fusion and classifier parameters."""
        yield from self.fusion.parameters()
        yield from self.global_fusion.parameters()
        yield from self.final_norm.parameters()
        yield from self.head.parameters()


def ugvnet_hybrid(
    num_classes: int,
    *,
    pretrained: bool = True,
    **kwargs,
) -> UGVNetHybrid:
    """Build the recommended EfficientNetV2-S + ConvNeXt-Tiny UGVNet."""
    return UGVNetHybrid(
        num_classes=num_classes,
        pretrained=pretrained,
        **kwargs,
    )
