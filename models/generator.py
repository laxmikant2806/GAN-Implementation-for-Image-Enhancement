# models/generator.py
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, nf=64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.BatchNorm2d(nf),
            nn.PReLU(),
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.BatchNorm2d(nf)
        )

    def forward(self, x):
        return x + self.block(x)

class UpsampleBlock(nn.Module):
    def __init__(self, nf, scale):
        super().__init__()
        self.conv = nn.Conv2d(nf, nf * scale**2, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        return self.prelu(self.pixel_shuffle(self.conv(x)))

class Generator(nn.Module):
    """16-block SRGAN generator (SRResNet backbone)."""
    def __init__(self, in_ch=1, out_ch=1, nf=64, num_res=16, upscale=4):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(in_ch, nf, 9, 1, 4),
            nn.PReLU()
        )

        self.res_blocks = nn.Sequential(*[ResidualBlock(nf) for _ in range(num_res)])

        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1)
        self.bn_trunk    = nn.BatchNorm2d(nf)

        up_layers = []
        for _ in range(int(upscale).bit_length() - 1):   # log2 scale
            up_layers += [UpsampleBlock(nf, 2)]
        self.upsampler = nn.Sequential(*up_layers)

        self.final = nn.Conv2d(nf, out_ch, 9, 1, 4)

    def forward(self, x):
        x1 = self.first(x)
        x2 = self.res_blocks(x1)
        x3 = self.bn_trunk(self.trunk_conv(x2))
        x = x1 + x3
        x = self.upsampler(x)
        return torch.tanh(self.final(x))
