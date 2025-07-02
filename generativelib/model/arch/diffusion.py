import torch.nn as nn
from diffusers.models.unets.unet_2d import UNet2DModel


class DiffusionUNet(nn.Module):
    def __init__(self, in_ch=1, f_base=64):
        super().__init__()
        self.model = UNet2DModel(
            in_channels=in_ch,
            out_channels=1,
            layers_per_block=2,
            block_out_channels=(f_base, f_base*2, f_base*2, f_base*4, f_base*4),
            down_block_types=("DownBlock2D",
                              "DownBlock2D",
                              "DownBlock2D",
                              "AttnDownBlock2D",
                              "DownBlock2D"),
            up_block_types=("UpBlock2D",
                            "AttnUpBlock2D",
                            "UpBlock2D",
                            "UpBlock2D",
                            "UpBlock2D",),
        )
        
    def forward(self, x, timesteps=None):
        return self.model(x, timesteps).sample