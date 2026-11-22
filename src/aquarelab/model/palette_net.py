import timm
import torch
import torch.nn.functional as F
from torch import nn


class ConvBlock(nn.Module):
    """Two 3x3 convolutions, each followed by batch norm and ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PaletteNet(nn.Module):
    """Pretrained encoder + U-Net style decoder that outputs K tone memberships per pixel."""

    def __init__(
        self, num_tones: int = 8, backbone: str = "resnet34", pretrained: bool = True
    ):
        super().__init__()
        self.num_tones = num_tones
        self.encoder = timm.create_model(
            backbone, features_only=True, pretrained=pretrained
        )
        channels = self.encoder.feature_info.channels()

        self.up_blocks = nn.ModuleList()
        in_ch = channels[-1]
        for skip_ch in reversed(channels[:-1]):
            self.up_blocks.append(ConvBlock(in_ch + skip_ch, skip_ch))
            in_ch = skip_ch
        self.head = nn.Conv2d(in_ch, num_tones, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        y = feats[-1]
        for block, skip in zip(self.up_blocks, reversed(feats[:-1])):
            y = F.interpolate(
                y, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
            y = block(torch.cat([y, skip], dim=1))
        y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(y).softmax(dim=1)  # (B, K, H, W), sums to 1 per pixel
