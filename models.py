"""models.py"""

import torch
from generative.networks.nets.diffusion_model_unet import DiffusionModelUNet

device = "cuda" if torch.cuda.is_available() else "cpu"

DiffusionModel = DiffusionModelUNet(
    spatial_dims=3,
    in_channels=3,
    out_channels=1,
    num_channels=(64, 64, 64),
    attention_levels=(False, False, True),
    num_res_blocks=1,
    num_head_channels=64,
    with_conditioning=False,
).to(device)