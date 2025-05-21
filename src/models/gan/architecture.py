from typing import List

import torch
from torch import nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
import torchvision.transforms.v2 as T

from src.models.gan.custom_transforms import OneOf, RandomRotate

class PartialConv2d(nn.Conv2d):
    """NVIDIA Realization"""
    def __init__(self, *args, **kwargs):

        # whether the mask is multi-channel or not
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False  

        if 'return_mask' in kwargs:
            self.return_mask = kwargs['return_mask']
            kwargs.pop('return_mask')
        else:
            self.return_mask = False

        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])
            
        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * self.weight_maskUpdater.shape[3]

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in = None):
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2], input.data.shape[3]).to(input)
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input)
                else:
                    mask = mask_in
                        
                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias = None, stride = self.stride, padding = self.padding, dilation = self.dilation, groups = 1)

                # for mixed precision training, change 1e-8 to 1e-6
                self.mask_ratio = self.slide_winsize/(self.update_mask + 1e-8)
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)


        raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask) if mask_in is not None else input)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)


        if self.return_mask:
            return output, self.update_mask
        else:
            return output


class PConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ks = 4, st = 2, pad = 1):
        super().__init__()
        self.p = PartialConv2d(in_ch, out_ch, ks, st, pad, bias = False, multi_channel = True, return_mask = True)
        self.a = nn.ReLU(True)
    
    def forward(self, x, m):
        x, m = self.p(x, m)
        return self.a(x), m


class ResidualPConv(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = PartialConv2d(ch, ch, 3, 1, 1, bias = False, multi_channel = True, return_mask = True)
        self.c2 = PartialConv2d(ch, ch, 3, 1, 1, bias = False, multi_channel = True, return_mask = True)
        self.a = nn.ReLU(True)
    
    def forward(self, x, m):
        y, m = self.c1(x, m)
        y = self.a(y)
        y, m = self.c2(y, m)
        return x + y, m


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout = False):
        super().__init__()
        self.u = nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias = False)
        self.a = nn.ReLU(True)
        self.d = nn.Dropout(0.5) if dropout else nn.Identity()
    
    def forward(self, x, m):
        x = self.u(x)
        x = self.a(x)
        x = self.d(x)
        m = F.interpolate(m, scale_factor = 2, mode = 'nearest')
        return x, m


class CustomGenerator(nn.Module):
    def __init__(self, in_ch = 1, f_base = 64):
        super().__init__()
        self.e1 = PConvBlock(in_ch, f_base)
        self.e2 = PConvBlock(f_base, f_base * 2)
        self.e3 = PConvBlock(f_base * 2, f_base * 4)
        self.e4 = PConvBlock(f_base * 4, f_base * 8)
        self.res = nn.Sequential(*[ResidualPConv(f_base * 8) for _ in range(3)])
        self.d4 = UpBlock(f_base * 16, f_base * 8, dropout=True)
        self.d3 = UpBlock(f_base * 12, f_base * 4)
        self.d2 = UpBlock(f_base * 6, f_base * 2)
        self.d1 = UpBlock(f_base * 3, f_base)
        self.o = nn.ConvTranspose2d(f_base, 1, 3, 1, 1)
    
    def forward(self, x):
        m = (x>0).float()
        e1, m1 = self.e1(x, m)
        e2, m2 = self.e2(e1, m1)
        e3, m3 = self.e3(e2, m2)
        e4, m4 = self.e4(e3, m3)
        b, mb = self.res[0](e4, m4)
        
        for blk in self.res[1:]:
            b, mb = blk(b, mb)
        
        d4, md5 = self.d4(torch.cat([b, e4], 1), mb)
        d3, _ = self.d3(torch.cat([d4, e3], 1), md5)
        d2, _ = self.d2(torch.cat([d3, e2], 1), m2)
        d1, _ = self.d1(torch.cat([d2, e1], 1), m1)
        
        return self.o(d1)

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
        ]


class CustomDiscriminator(nn.Module):
    def __init__(self, in_ch = 1, f_base = 64):
        super(CustomDiscriminator, self).__init__()
        
        def conv_block(in_ch, out_ch, kernel_size = 4, stride = 2, padding = 1):
            layers = [nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias = True)]
            layers.append(nn.LeakyReLU(0.2, inplace = True))
            return nn.Sequential(*layers)
        
        self.net = nn.Sequential(
            conv_block(in_ch, f_base), 
            conv_block(f_base, f_base * 2), 
            conv_block(f_base * 2, f_base * 4), 
            conv_block(f_base * 4, f_base * 8), 
            nn.Conv2d(f_base * 8, 1, kernel_size = 3, stride = 1, padding = 1, bias = False), 
        )
    
    def forward(self, x):
        out = self.net(x)
        return out
