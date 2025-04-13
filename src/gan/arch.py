import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_instancenorm=True):
        super(ConvBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True) if use_instancenorm else nn.Identity(),
            nn.ReLU(inplace=True)
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(channels, affine=True)
        )

    def forward(self, x):
        return x + self.block(x)

class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(UpConvBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        layers = [
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)

class WGanGenerator(nn.Module):
    def __init__(self, input_channels=2, feature_maps=64):
        super(WGanGenerator, self).__init__()

        self.enc1 = ConvBlock(input_channels, feature_maps, use_instancenorm=False)
        self.enc2 = ConvBlock(feature_maps, feature_maps * 2)
        self.enc3 = ConvBlock(feature_maps * 2, feature_maps * 4)
        self.enc4 = ConvBlock(feature_maps * 4, feature_maps * 8)
        self.enc5 = ConvBlock(feature_maps * 8, feature_maps * 8)

        self.bottleneck = nn.Sequential(
            *[ResidualBlock(feature_maps * 8) for _ in range(3)]
        )

        self.dec5 = UpConvBlock(feature_maps * 16, feature_maps * 8, dropout=True)  # 512+512 -> 1024
        self.dec4 = UpConvBlock(feature_maps * 16, feature_maps * 4, dropout=True)  # 512+512 -> 1024
        self.dec3 = UpConvBlock(feature_maps * 8, feature_maps * 2, dropout=False)  # 256+256 -> 512
        self.dec2 = UpConvBlock(feature_maps * 4, feature_maps, dropout=False)     # 128+128 -> 256
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(feature_maps, affine=True),
            nn.ReLU(inplace=True)
        )
        
        self.final = nn.Sequential(
            nn.ConvTranspose2d(feature_maps, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x, mask):
        x_in = torch.cat([x, mask], dim=1)
        
        e1 = self.enc1(x_in)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        bn = self.bottleneck(e5)

        d5 = self.dec5(torch.cat([bn, e5], dim=1))
        d4 = self.dec4(torch.cat([d5, e4], dim=1))
        d3 = self.dec3(torch.cat([d4, e3], dim=1))
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))

        output = self.final(d1)
        return output


class WGanCritic(nn.Module):
    def __init__(self, input_channels=1, feature_maps=64):
        super(WGanCritic, self).__init__()
                
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, feature_maps, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(feature_maps, feature_maps * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(feature_maps * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(feature_maps * 2, feature_maps * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(feature_maps * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(feature_maps * 4, feature_maps * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(feature_maps * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.final = nn.Conv2d(feature_maps * 8, 1, kernel_size=1, stride=1, padding=0)
             
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.final(x)
        return x