"""main.py"""

import sys
import wandb
import torch

from train import train_diffusion
from data import get_dataloader
from models import DiffusionModel
from config import (NUM_EPOCHS, INPUT_DIR, BATCH_SIZE)

# wandb login
WANDB_API_KEY = "8b67af0ea5e8251ee45c6180b5132d513b68c079"  
wandb.login(key=WANDB_API_KEY)

# create dataloader
train_dataloader, validation_dataloader = get_dataloader(INPUT_DIR, batch_size=BATCH_SIZE)

# init  wandb run
wandb.init(project="BraTS-inPainting-Diffusuin-2025")

# run training
trained_G, trained_D = train_diffusion(DiffusionModel, train_dataloader, validation_dataloader, NUM_EPOCHS)
