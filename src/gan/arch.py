import random
from typing import List
import torch
import torch.nn as nn
import torchvision.transforms.v2 as T

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super(ConvBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=False) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True)
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(channels, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(channels, affine=False),
        )

    def forward(self, x):
        return x + self.block(x)

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False, use_se=True):
        super(UpConvBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        layers = [
            nn.InstanceNorm2d(out_channels, affine=False),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.conv = nn.Sequential(*layers)
        self.se = SEBlock(out_channels) if use_se else nn.Identity()

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return self.se(x)

class WGanGenerator(nn.Module):
    def __init__(self, input_channels=1, feature_maps=64):
        super(WGanGenerator, self).__init__()

        self.enc1 = ConvBlock(input_channels, feature_maps, use_batchnorm=False)
        self.enc2 = ConvBlock(feature_maps, feature_maps * 2)
        self.enc3 = ConvBlock(feature_maps * 2, feature_maps * 4)
        self.enc4 = ConvBlock(feature_maps * 4, feature_maps * 8)
        self.enc5 = ConvBlock(feature_maps * 8, feature_maps * 16)

        self.bottleneck = nn.Sequential(
            *[ResidualBlock(feature_maps * 16) for _ in range(3)]
        )

        self.dec5 = UpConvBlock(feature_maps * 16 + feature_maps * 16, feature_maps * 8, dropout=True)
        self.dec4 = UpConvBlock(feature_maps * 8 + feature_maps * 8, feature_maps * 8, dropout=True)
        self.dec3 = UpConvBlock(feature_maps * 8 + feature_maps * 4, feature_maps * 4, dropout=False)
        self.dec2 = UpConvBlock(feature_maps * 4 + feature_maps * 2, feature_maps * 2, dropout=False)
        self.dec1 = UpConvBlock(feature_maps * 2 + feature_maps, feature_maps, dropout=False)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(feature_maps, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        e1 = self.enc1(x)
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
    
    @staticmethod
    def get_train_transforms(target_img_size) -> List[T.Transform]:
        max_crop = 1024
        return [
            T.ToImage(),
            T.Resize((max_crop, max_crop), interpolation=T.InterpolationMode.BILINEAR),
            OneOf([T.RandomCrop((size, size)) for size in range(640, max_crop, 128)], p=1.0),

            RandomRotate(p=0.8),
            T.RandomHorizontalFlip(p=0.8),
            T.RandomVerticalFlip(p=0.8),

            T.Resize((target_img_size, target_img_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToDtype(torch.float32, scale=True),
        ]
    
    @staticmethod
    def get_input_transforms(target_img_size) -> 'List[T.Transform]':
        return [
            T.ToImage(),
            T.Resize((target_img_size, target_img_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToDtype(torch.float32, scale=True)
        ]


class WGanCritic(nn.Module):
    def __init__(self, input_channels=1, feature_maps=64):
        super(WGanCritic, self).__init__()
        
        def conv_block(in_ch, out_ch, kernel_size=4, stride=2, padding=1, use_in=False):
            layers = [nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=True)]
            if use_in:
                layers.append(nn.InstanceNorm2d(out_ch, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)
        
        self.net = nn.Sequential(
            conv_block(input_channels, feature_maps, use_in=False),
            conv_block(feature_maps, feature_maps*2, use_in=False),
            conv_block(feature_maps*2, feature_maps*4, use_in=False),
            conv_block(feature_maps*4, feature_maps*8, use_in=False),
            nn.Conv2d(feature_maps*8, 1, kernel_size=4, stride=1, padding=0, bias=False)
        )
    
    def forward(self, x):
        out = self.net(x)
        return out


class OneOf(torch.nn.Module):
    def __init__(self, transforms, p=1.0):
        super().__init__()
        self.transforms = transforms
        self.p = p

    def forward(self, x):
        if random.random() < self.p:
            t = random.choice(self.transforms)
            return t(x)
        return x

 
class RandomRotate(torch.nn.Module):
    def __init__(self, angles=(90, 180, 270), p=0.8):
        super().__init__()
        self.angles = angles
        self.p = p

    def forward(self, x):
        if random.random() < self.p:
            angle = random.choice(self.angles)
            return T.functional.rotate(x, angle)
        return x
