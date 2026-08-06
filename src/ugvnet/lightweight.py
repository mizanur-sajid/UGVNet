"""PyTorch implementation of UGVNet.

UGVNet is a hierarchical hybrid vision backbone. Early convolutional stages
preserve local texture and edge information, while later UGV blocks combine a
depthwise-convolutional local path with global multi-head self-attention.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import Tensor, nn


class DropPath(nn.Module):
    """Stochastic depth applied independently to each sample."""

    def __init__(self, probability: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= probability < 1.0:
            raise ValueError("DropPath probability must be in [0, 1).")
        self.probability = probability

    def forward(self, x: Tensor) -> Tensor:
        if self.probability == 0.0 or not self.training:
            return x
        keep_probability = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_probability)
        return x * mask.div(keep_probability)


class LayerNorm2d(nn.Module):
    """Layer normalization over channels for NCHW feature maps."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class ConvNormAct(nn.Sequential):
    """Convolution followed by batch normalization and optional activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        activation: bool = True,
    ) -> None:
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if activation:
            layers.append(nn.GELU())
        super().__init__(*layers)


class SqueezeExcite(nn.Module):
    """Channel attention used in the local convolutional path."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden_channels = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.reduce = nn.Conv2d(channels, hidden_channels, 1)
        self.act = nn.SiLU()
        self.expand = nn.Conv2d(hidden_channels, channels, 1)
        self.gate = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        scale = self.pool(x)
        scale = self.expand(self.act(self.reduce(scale)))
        return x * self.gate(scale)


class MBConv(nn.Module):
    """Inverted residual block for efficient local feature extraction."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: float = 4.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_channels = round(in_channels * expansion)
        self.use_residual = stride == 1 and in_channels == out_channels
        self.expand = ConvNormAct(in_channels, hidden_channels, kernel_size=1)
        self.depthwise = ConvNormAct(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            stride=stride,
            groups=hidden_channels,
        )
        self.se = SqueezeExcite(hidden_channels)
        self.project = ConvNormAct(
            hidden_channels, out_channels, kernel_size=1, activation=False
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x: Tensor) -> Tensor:
        output = self.project(self.se(self.depthwise(self.expand(x))))
        if self.use_residual:
            output = x + self.drop_path(output)
        return output


class LocalMixer(nn.Module):
    """Local spatial mixing path inside a unified UGV block."""

    def __init__(self, channels: int, expansion: float = 2.0) -> None:
        super().__init__()
        hidden_channels = int(channels * expansion)
        self.expand = nn.Conv2d(channels, hidden_channels, 1)
        self.depthwise = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
            groups=hidden_channels,
        )
        self.act = nn.GELU()
        self.se = SqueezeExcite(hidden_channels)
        self.project = nn.Conv2d(hidden_channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.expand(x)
        x = self.act(self.depthwise(x))
        return self.project(self.se(x))


class GlobalSelfAttention(nn.Module):
    """Global self-attention over all spatial positions."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by num_heads ({num_heads})."
            )
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.projection = nn.Conv2d(channels, channels, kernel_size=1)
        self.projection_dropout = nn.Dropout(projection_dropout)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, channels, height, width = x.shape
        tokens = height * width
        if tokens > 10000:
            raise RuntimeError(f"Attention resolution too large ({height}x{width}). Max supported tokens is 10,000 to prevent OOM.")
        
        qkv = self.qkv(x).reshape(batch_size, 3, self.num_heads, self.head_dim, tokens)
        query, key, value = qkv.unbind(dim=1)
        
        query = query.transpose(-2, -1)
        key = key.transpose(-2, -1)
        value = value.transpose(-2, -1)
        
        dropout_p = self.attention_dropout.p if self.training else 0.0
        output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, dropout_p=dropout_p
        )
        output = output.transpose(-2, -1).reshape(batch_size, channels, height, width)
        return self.projection_dropout(self.projection(output))


class ConvMLP(nn.Module):
    """Channel MLP with a depthwise convolution for spatial continuity."""

    def __init__(
        self, channels: int, expansion: float = 4.0, dropout: float = 0.0
    ) -> None:
        super().__init__()
        hidden_channels = int(channels * expansion)
        self.layers = nn.Sequential(
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

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class UGVBlock(nn.Module):
    """Unified block that sequentially mixes local and global information."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        layer_scale_init: float = 1e-5,
    ) -> None:
        super().__init__()
        self.local_norm = LayerNorm2d(channels)
        self.local_mixer = LocalMixer(channels)
        self.global_norm = LayerNorm2d(channels)
        self.global_mixer = GlobalSelfAttention(
            channels,
            num_heads,
            attention_dropout=attention_dropout,
            projection_dropout=dropout,
        )
        self.mlp_norm = LayerNorm2d(channels)
        self.mlp = ConvMLP(channels, expansion=mlp_ratio, dropout=dropout)
        self.drop_path = DropPath(drop_path)
        self.local_scale = nn.Parameter(torch.full((channels, 1, 1), layer_scale_init))
        self.global_scale = nn.Parameter(torch.full((channels, 1, 1), layer_scale_init))
        self.mlp_scale = nn.Parameter(torch.full((channels, 1, 1), layer_scale_init))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop_path(self.local_scale * self.local_mixer(self.local_norm(x)))
        x = x + self.drop_path(
            self.global_scale * self.global_mixer(self.global_norm(x))
        )
        x = x + self.drop_path(self.mlp_scale * self.mlp(self.mlp_norm(x)))
        return x


class LocalStage(nn.Sequential):
    """A downsampling MBConv stage."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        drop_paths: Sequence[float],
    ) -> None:
        blocks: list[nn.Module] = []
        for index in range(depth):
            blocks.append(
                MBConv(
                    in_channels if index == 0 else out_channels,
                    out_channels,
                    stride=2 if index == 0 else 1,
                    drop_path=drop_paths[index],
                )
            )
        super().__init__(*blocks)


class UGVStage(nn.Module):
    """A downsampling stage followed by unified local-global blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        num_heads: int,
        drop_paths: Sequence[float],
        mlp_ratio: float,
        dropout: float,
        attention_dropout: float,
        layer_scale_init: float,
    ) -> None:
        super().__init__()
        self.downsample = nn.Sequential(
            LayerNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
        )
        # Conditional positional encoding works at any input resolution.
        self.position = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            groups=out_channels,
        )
        self.blocks = nn.Sequential(
            *[
                UGVBlock(
                    out_channels,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_paths[index],
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    layer_scale_init=layer_scale_init,
                )
                for index in range(depth)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.downsample(x)
        x = x + self.position(x)
        return self.blocks(x)


@dataclass(frozen=True)
class UGVNetConfig:
    """Architecture configuration for a UGVNet variant."""

    widths: tuple[int, int, int, int, int]
    depths: tuple[int, int, int, int]
    heads: tuple[int, int]
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    attention_dropout: float = 0.0
    drop_path_rate: float = 0.1
    layer_scale_init: float = 1e-5

    def __post_init__(self) -> None:
        if len(self.widths) != 5 or len(self.depths) != 4 or len(self.heads) != 2:
            raise ValueError(
                "UGVNetConfig requires 5 widths, 4 depths, and 2 head counts."
            )
        if any(width < 1 for width in self.widths):
            raise ValueError("Every stage width must be positive.")
        if any(depth < 1 for depth in self.depths):
            raise ValueError("Every stage must contain at least one block.")
        if any(heads < 1 for heads in self.heads):
            raise ValueError("Every attention head count must be positive.")
        for channels, heads in zip(self.widths[-2:], self.heads):
            if channels % heads != 0:
                raise ValueError(
                    f"Stage width {channels} must be divisible by {heads} heads."
                )
        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive.")
        for name, value in (
            ("dropout", self.dropout),
            ("attention_dropout", self.attention_dropout),
            ("drop_path_rate", self.drop_path_rate),
        ):
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1).")
        if self.layer_scale_init < 0:
            raise ValueError("layer_scale_init must be non-negative.")


MODEL_CONFIGS: dict[str, UGVNetConfig] = {
    "tiny": UGVNetConfig(
        widths=(32, 48, 96, 192, 320),
        depths=(1, 2, 2, 2),
        heads=(6, 10),
        drop_path_rate=0.10,
    ),
    "small": UGVNetConfig(
        widths=(32, 64, 128, 256, 384),
        depths=(2, 2, 4, 3),
        heads=(8, 12),
        drop_path_rate=0.20,
    ),
    "base": UGVNetConfig(
        widths=(48, 96, 192, 384, 512),
        depths=(2, 3, 6, 4),
        heads=(12, 16),
        drop_path_rate=0.30,
    ),
}


class UGVNet(nn.Module):
    """Unified Global Vision Network for image classification.

    The network returns class logits from ``forward`` and exposes spatial
    features through ``forward_features`` and ``forward_intermediates``.
    """

    def __init__(
        self,
        config: UGVNetConfig,
        num_classes: int = 1000,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        if num_classes < 0:
            raise ValueError("num_classes must be non-negative.")
        if in_channels < 1:
            raise ValueError("in_channels must be positive.")
        self.config = config
        self.num_classes = num_classes
        self.feature_channels = config.widths[1:]
        widths = config.widths

        self.stem = nn.Sequential(
            ConvNormAct(in_channels, widths[0], kernel_size=3, stride=2),
            ConvNormAct(
                widths[0],
                widths[0],
                kernel_size=3,
                groups=widths[0],
            ),
        )

        total_blocks = sum(config.depths)
        drop_paths = torch.linspace(0, config.drop_path_rate, total_blocks).tolist()
        offset = 0

        local_stages: list[nn.Module] = []
        for stage_index in range(2):
            depth = config.depths[stage_index]
            local_stages.append(
                LocalStage(
                    widths[stage_index],
                    widths[stage_index + 1],
                    depth,
                    drop_paths[offset : offset + depth],
                )
            )
            offset += depth
        self.local_stages = nn.ModuleList(local_stages)

        global_stages: list[nn.Module] = []
        for global_index in range(2):
            stage_index = global_index + 2
            depth = config.depths[stage_index]
            global_stages.append(
                UGVStage(
                    widths[stage_index],
                    widths[stage_index + 1],
                    depth,
                    num_heads=config.heads[global_index],
                    drop_paths=drop_paths[offset : offset + depth],
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout,
                    attention_dropout=config.attention_dropout,
                    layer_scale_init=config.layer_scale_init,
                )
            )
            offset += depth
        self.global_stages = nn.ModuleList(global_stages)

        self.final_norm = LayerNorm2d(widths[-1])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head_dropout = nn.Dropout(config.dropout)
        self.head: nn.Module = (
            nn.Linear(widths[-1], num_classes) if num_classes > 0 else nn.Identity()
        )
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward_intermediates(self, x: Tensor) -> tuple[Tensor, ...]:
        """Return feature maps at strides 4, 8, 16, and 32."""
        features: list[Tensor] = []
        x = self.stem(x)
        for stage in self.local_stages:
            x = stage(x)
            features.append(x)
        for stage in self.global_stages:
            x = stage(x)
            features.append(x)
        return tuple(features)

    def forward_features(self, x: Tensor) -> Tensor:
        """Return the final normalized spatial feature map."""
        return self.final_norm(self.forward_intermediates(x)[-1])

    def forward_head(self, features: Tensor) -> Tensor:
        """Pool a feature map and apply the classification head."""
        features = self.pool(features).flatten(1)
        return self.head(self.head_dropout(features))

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_head(self.forward_features(x))

    def reset_classifier(self, num_classes: int) -> None:
        """Replace the classifier while retaining the learned backbone."""
        if num_classes < 0:
            raise ValueError("num_classes must be non-negative.")
        self.num_classes = num_classes
        if num_classes > 0:
            reference = next(self.parameters())
            self.head = nn.Linear(self.config.widths[-1], num_classes).to(
                device=reference.device,
                dtype=reference.dtype,
            )
        else:
            self.head = nn.Identity()
        self._initialize_weights(self.head)


def create_ugvnet(
    variant: str = "tiny",
    *,
    num_classes: int = 1000,
    in_channels: int = 3,
    **config_overrides: Any,
) -> UGVNet:
    """Create a named UGVNet variant with optional configuration overrides."""
    normalized_variant = variant.lower().removeprefix("ugvnet_")
    if normalized_variant not in MODEL_CONFIGS:
        choices = ", ".join(sorted(MODEL_CONFIGS))
        raise ValueError(f"Unknown UGVNet variant '{variant}'. Choose from: {choices}.")
    config = MODEL_CONFIGS[normalized_variant]
    if config_overrides:
        valid_fields = set(config.__dataclass_fields__)
        unknown = set(config_overrides) - valid_fields
        if unknown:
            raise TypeError(f"Unknown configuration overrides: {sorted(unknown)}")
        config = replace(config, **config_overrides)
    return UGVNet(config, num_classes=num_classes, in_channels=in_channels)


def ugvnet_tiny(num_classes: int = 1000, in_channels: int = 3) -> UGVNet:
    return create_ugvnet("tiny", num_classes=num_classes, in_channels=in_channels)


def ugvnet_small(num_classes: int = 1000, in_channels: int = 3) -> UGVNet:
    return create_ugvnet("small", num_classes=num_classes, in_channels=in_channels)


def ugvnet_base(num_classes: int = 1000, in_channels: int = 3) -> UGVNet:
    return create_ugvnet("base", num_classes=num_classes, in_channels=in_channels)
