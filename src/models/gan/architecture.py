from typing import List

import torch
from torch import nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
import torchvision.transforms.v2 as T

from src.models.gan.custom_transforms import OneOf, RandomRotate, BinarizeTransform


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True),
            nn.Dropout(.25),
            nn.ReLU(inplace=True)
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(True),
            nn.Dropout(.5),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(True),
        )

    def forward(self, x):
        return x + self.block(x)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout = False):
        super().__init__()
        self.u = nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias = False)
        self.a = nn.ReLU(True)
        self.d = nn.Dropout(0.25) if dropout else nn.Identity()
    
    def forward(self, x, m):
        x = self.u(x)
        x = self.a(x)
        x = self.d(x)
        m = F.interpolate(m, scale_factor = 2, mode = 'nearest')
        return x, m


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.u = nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias = False)
        self.a = nn.ReLU(True)
        self.se = SEBlock(out_ch)
    
    def forward(self, x):
        x = self.u(x)
        x = self.a(x)
        
        x = self.se(x)
        return x


class CustomGenerator(nn.Module):
    def __init__(self, in_ch=1, f_base=64):
        super(CustomGenerator, self).__init__()

        self.enc1 = ConvBlock(in_ch, f_base)
        self.enc2 = ConvBlock(f_base, f_base * 2)
        self.enc3 = ConvBlock(f_base * 2, f_base * 4)
        self.enc4 = ConvBlock(f_base * 4, f_base * 8)

        self.bottleneck = nn.Sequential(
            *[ResidualBlock(f_base * 8) for _ in range(3)]
        )

        self.dec4 = UpBlock(f_base * 8 + f_base * 8, f_base * 8)
        self.dec3 = UpBlock(f_base * 8 + f_base * 4, f_base * 4)
        self.dec2 = UpBlock(f_base * 4 + f_base * 2, f_base * 2)
        self.dec1 = UpBlock(f_base * 2 + f_base, f_base)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(f_base, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        bn = self.bottleneck(e4)

        d4 = self.dec4(torch.cat([bn, e4], dim=1))
        d3 = self.dec3(torch.cat([d4, e3], dim=1))
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))

        output = self.final(d1)
        return output

    @staticmethod
    def get_transforms(target_img_size) -> List[T.Transform]:
        max_crop = 1024
        return [
            T.ToImage(), 
            T.Resize((max_crop, max_crop), interpolation = T.InterpolationMode.BILINEAR), 
            OneOf([T.RandomCrop((size, size)) for size in range(640, max_crop, 128)], p = 1.0), 

            RandomRotate(p = 0.8), 
            T.RandomHorizontalFlip(p = 0.8), 
            T.RandomVerticalFlip(p = 0.8), 

            T.Resize((target_img_size, target_img_size), interpolation = T.InterpolationMode.BILINEAR), 
            T.ToDtype(torch.float32, scale = True), 
            BinarizeTransform()
        ]
    
    @staticmethod
    def get_infer_transforms(target_img_size) -> List[T.Transform]:
        return [
            T.ToImage(),
            T.Resize((target_img_size, target_img_size), interpolation = T.InterpolationMode.BILINEAR), 
            T.ToDtype(torch.float32, scale = True), 
            BinarizeTransform()
        ]

class CustomDiscriminator(nn.Module):
    def __init__(self, in_ch = 1, f_base = 64):
        super(CustomDiscriminator, self).__init__()
        
        def conv_block(in_ch, out_ch, kernel_size = 4, stride = 2, padding = 1):
            layers = [spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias = True))]
            layers.append(nn.LeakyReLU(0.2, inplace = True))
            return nn.Sequential(*layers)
        
        self.net = nn.Sequential(
            conv_block(in_ch, f_base), 
            conv_block(f_base, f_base * 2), 
            conv_block(f_base * 2, f_base * 4), 
            conv_block(f_base * 4, f_base * 8), 
            spectral_norm(nn.Conv2d(f_base * 8, 1, kernel_size = 3, stride = 1, padding = 1, bias = True)),
        )
    
    def forward(self, x):
        out = self.net(x)
        return out
