import torch
from torch import nn


import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = self.query_conv(x).view(B, -1, H*W).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(B, -1, H*W)
        proj_value = self.value_conv(x).view(B, -1, H*W)
        
        energy = torch.bmm(proj_query, proj_key)
        attention = torch.softmax(energy, dim=-1)
        
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        out = self.gamma * out + x
        
        return out

class GanGenerator(nn.Module):
    def __init__(self, in_ch, f_base, n_down=4, n_residual=2, max_channels=512):
        super().__init__()

        def capped_ch(ch):
            return min(ch, max_channels)

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                nn.LeakyReLU(inplace=True)
            )

        def residual_block(channels, dropout=0.0):
            layers = [
                nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.LeakyReLU(inplace=True)
            ]
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            
            layers.append(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False))
            return nn.Sequential(*layers)

        def up_block(in_ch, out_ch):
            return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
                nn.LeakyReLU(inplace=True)
            )

        # Энкодер
        self.encoders = nn.ModuleList()
        ch = in_ch
        enc_channels = []
        for i in range(n_down):
            out_ch = capped_ch(f_base * (2 ** i))
            self.encoders.append(conv_block(ch, out_ch))
            enc_channels.append(out_ch)
            ch = out_ch

        # Боттлнек
        self.bottleneck = nn.Sequential(*[residual_block(ch) for _ in range(n_residual)])

        # Self-attention
        self.attn = SelfAttention(ch)

        # Декодер
        self.decoders = nn.ModuleList()
        in_ch_dec = enc_channels[-1]
        for out_ch in reversed(enc_channels):
            self.decoders.append(up_block(in_ch_dec + out_ch, out_ch))
            in_ch_dec = out_ch

        # Финальный слой
        self.final = nn.Sequential(
            nn.ConvTranspose2d(f_base, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Tanh()
        )

    def forward(self, x):
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)

        x = self.bottleneck(x)
        x = self.attn(x)

        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(torch.cat([x, skip], dim=1))

        return self.final(x)



class GanDiscriminator(nn.Module):
    def __init__(self, in_ch, f_base, max_channels=512, layers_count=6):
        super().__init__()
        
        def layer(in_channels, out_channels, k=4, s=2, p=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, k, s, p),
                nn.LeakyReLU(0.2, inplace=True)
            )

        layers = []
        ch_in = in_ch
        ch_out = f_base
        for _ in range(layers_count):
            layers.append(layer(ch_in, ch_out))
            ch_in = ch_out
            ch_out = min(ch_out * 2, max_channels)

        layers.append(nn.Conv2d(ch_in, 1, 4, 1, 0))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return out.view(x.size(0))
