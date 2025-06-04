import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels, dropout=0.5):
        super().__init__()
        layers = [
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True)
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x)

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.se = SEBlock(out_ch)

    def forward(self, x):
        x = self.up(x)
        return self.se(x)

class GanGenerator(nn.Module):
    def __init__(self, in_ch, f_base, n_down=4, n_residual=3):
        super().__init__()
        # Энкодер
        self.encoders = nn.ModuleList()
        ch = in_ch
        features_limit = 2048
        
        for i in range(n_down):
            out_ch = min(f_base * (2**i), features_limit)
            self.encoders.append(ConvBlock(ch, out_ch))
            ch = out_ch

        # Боттлнек
        self.bottleneck = nn.Sequential(
            *[ResidualBlock(channels=ch) for _ in range(n_residual)]
        )

        # Декодер
        self.decoders = nn.ModuleList()
        enc_channels = [min(f_base * (2**i), features_limit) for i in range(n_down)]
        in_ch_dec = enc_channels[-1]
        for out_ch in reversed(enc_channels):
            self.decoders.append(UpBlock(in_ch_dec + out_ch, out_ch))
            in_ch_dec = out_ch

        # Финальный слой
        self.final = nn.Sequential(
            nn.ConvTranspose2d(f_base, 1, kernel_size=3, stride=1, padding=1, bias=True),
        )

    def forward(self, x):
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)

        x = self.bottleneck(x)

        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(torch.cat([x, skip], dim=1))

        return self.final(x)


class GanDiscriminator(nn.Module):
    def __init__(self, in_ch, f_base, layers_count: int=6):
        super().__init__()
        out_ch = 1
        features_limit = 2048
        
        def layer(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.LeakyReLU(0.2, inplace=True)
            )
        
        def make_layers(in_channels, base_channels, count):
            blocks = []
            for i in range(count):
                out_channels = min(base_channels * (2 ** i), features_limit)
                blocks.append(layer(in_channels, out_channels))
                in_channels = out_channels
            return nn.Sequential(*blocks)
        
        self.net = make_layers(in_ch, f_base, layers_count)
        
        self.final = nn.Conv2d(min(f_base * 2**(layers_count-1), features_limit), out_ch, kernel_size=4, stride=1, padding=0, bias=False)
    
    def forward(self, x):
        x = self.net(x)
        return self.final(x).view(x.size(0), -1)
