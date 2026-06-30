"""
Compact CNN for Signal Spectrogram Classification — Run 16.

Built entirely from PyTorch primitives as required by competition rules.

Key insight: Images are viridis-colormap-encoded scalars. We use single-channel
grayscale input at native resolution (128×64) with a compact ~800K param model.

Architecture:
    Input: [B, 1, 128, 64]

    Stem: Conv(1→32, 3×3, pad=1) → BN → SiLU                [128×64]

    Block 1: Conv(32→64, 3×3, stride=2, pad=1) → BN → SiLU → SE(64)    [64×32]
    Block 1b: Conv(64→64, 3×3, pad=1) → BN → SiLU                       [64×32] (residual)

    Block 2: Conv(64→128, 3×3, stride=2, pad=1) → BN → SiLU → SE(128)  [32×16]
    Block 2b: Conv(128→128, 3×3, pad=1) → BN → SiLU                     [32×16] (residual)

    Block 3: Conv(128→256, 3×3, stride=2, pad=1) → BN → SiLU → SE(256) [16×8]
    Block 3b: Conv(256→256, 3×3, pad=1) → BN → SiLU                     [16×8] (residual)

    Block 4: Conv(256→256, 3×3, stride=2, pad=1) → BN → SiLU → SE(256) [8×4]

    Head: AdaptiveAvgPool2d(1) → Flatten → [B, 256]
          → Linear(256→128) → SiLU → Dropout(0.5) → Linear(128→5)

Total parameters: ~900K
"""
import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        squeezed = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, squeezed, bias=False),
            nn.SiLU(),
            nn.Linear(squeezed, channels, bias=False),
            nn.Sigmoid(),
        )
        self._channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        scale = self.pool(x).view(b, -1)
        scale = self.fc(scale).view(b, self._channels, 1, 1)
        return x * scale


class ConvBNSiLU(nn.Module):
    """Conv → BatchNorm → SiLU activation."""
    def __init__(self, in_c: int, out_c: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, groups: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride=stride,
                              padding=padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    """Residual block: Conv→BN→SiLU + skip connection."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)
        # Zero-init the last BN so block starts as identity
        nn.init.zeros_(self.bn2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + x)


class CompactSignalCNN(nn.Module):
    """Compact CNN for single-channel spectrogram classification."""
    def __init__(self, num_classes: int = 5, dropout_p: float = 0.5):
        super().__init__()

        # Stem: [1, 128, 64] → [32, 128, 64]
        self.stem = ConvBNSiLU(1, 32, 3, stride=1, padding=1)

        # Downsampling blocks with SE attention + residual refinement
        # Block 1: [32, 128, 64] → [64, 64, 32]
        self.down1 = ConvBNSiLU(32, 64, 3, stride=2, padding=1)
        self.se1 = SEBlock(64)
        self.res1 = ResBlock(64)

        # Block 2: [64, 64, 32] → [128, 32, 16]
        self.down2 = ConvBNSiLU(64, 128, 3, stride=2, padding=1)
        self.se2 = SEBlock(128)
        self.res2 = ResBlock(128)

        # Block 3: [128, 32, 16] → [256, 16, 8]
        self.down3 = ConvBNSiLU(128, 256, 3, stride=2, padding=1)
        self.se3 = SEBlock(256)
        self.res3 = ResBlock(256)

        # Block 4: [256, 16, 8] → [256, 8, 4]
        self.down4 = ConvBNSiLU(256, 256, 3, stride=2, padding=1)
        self.se4 = SEBlock(256)

        # Head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None and m.weight.requires_grad:
                    # Don't overwrite the zero-init in ResBlock.bn2
                    pass
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        x = self.down1(x)
        x = self.se1(x)
        x = self.res1(x)

        x = self.down2(x)
        x = self.se2(x)
        x = self.res2(x)

        x = self.down3(x)
        x = self.se3(x)
        x = self.res3(x)

        x = self.down4(x)
        x = self.se4(x)

        return self.head(x)


if __name__ == '__main__':
    model = CompactSignalCNN(num_classes=5)
    total = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {total:,} ({total/1e6:.2f}M)')
    x = torch.randn(2, 1, 128, 64)
    out = model(x)
    print(f'Input: {x.shape} → Output: {out.shape}')
