"""config.py"""

import os
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# paths
INPUT_DIR = "/kaggle/working/diffusion-inpainting/ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge-Training"

# training parameters
BATCH_SIZE = 1
NUM_EPOCHS = 350

# loss weights
LAMBDA_L1 = 1.0
LAMBDA_PERCEPT = 1.0
LAMBDA_SSIM = 2.5
LAMBDA_PSNR = 0.5

# directories
WEIGHT_DIR = "weight"
GENERATED_DIR = "generated"
LOG_FILE = "train.pkl"

os.makedirs(WEIGHT_DIR, exist_ok=True)
os.makedirs(GENERATED_DIR, exist_ok=True)

# Wandb configuration
WANDB_PROJECT = "Pix2Pix-MRI-Style-Transfer"