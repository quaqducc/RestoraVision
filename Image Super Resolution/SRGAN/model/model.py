import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    # Conv -> BN -> Leaky/PReLu
    def __init__(
            self,
            in_channels,
            out_channels,
            discriminator=False,
            use_act=True,
            use_bn=True,
            **kwargs,
    ):
        super().__init__()
        self.cnn = nn.Conv2d(in_channels, out_channels, **kwargs, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.use_act = use_act
        self.act = (
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            if discriminator
            else nn.PReLU(num_parameters=out_channels)
        )

    def forward(self, x):
        out = self.bn(self.cnn(x))
        return self.act(out) if self.use_act else out


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels * scale_factor ** 2,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.ps = nn.PixelShuffle(scale_factor)  # in_channels * 4, H, W --> in_channels, H * 2, W * 2
        self.act = nn.PReLU(num_parameters=in_channels)

    def forward(self, x):
        return self.act(self.ps(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block1 = ConvBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            use_act=True,
            use_bn=True
        )
        self.block2 = ConvBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            use_act=False,
            use_bn=True
        )

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        return out + x


class Generator(nn.Module):
    def __init__(self, in_channels=3, num_channels=64, num_blocks=16):
        super().__init__()
        self.initial = ConvBlock(
            in_channels=in_channels,
            out_channels=num_channels,
            kernel_size=9,
            stride=1,
            padding=4,
            use_act=True,
            use_bn=False
        )
        self.residuals = nn.Sequential(*[ResidualBlock(num_channels) for _ in range(num_blocks)])
        self.conv_block = ConvBlock(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            use_act=False,
            use_bn=True
        )
        self.upsamples = nn.Sequential(UpsampleBlock(num_channels, 2), UpsampleBlock(num_channels, 2))
        self.final = nn.Conv2d(
            in_channels=num_channels,
            out_channels=in_channels,
            kernel_size=9,
            stride=1,
            padding=4
        )

    def forward(self, x):
        initial = self.initial(x)
        x = self.residuals(initial)
        x = self.conv_block(x) + initial
        x = self.upsamples(x)
        return torch.tanh(self.final(x))


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 64, 128, 128, 256, 256, 512, 512]):
        super().__init__()
        blocks = []
        for idx, feature in enumerate(features):
            blocks.append(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=feature,
                    kernel_size=3,
                    stride=1 + idx % 2,
                    padding=1,
                    discriminator=True,
                    use_act=True,
                    use_bn=False if idx == 0 else True
                )
            )
            in_channels = feature

        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(in_features=512 * 6 * 6, out_features=1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(in_features=1024, out_features=1),
        )

    def forward(self, x):
        x = self.blocks(x)
        return self.classifier(x)
