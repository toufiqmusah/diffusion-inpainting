"""data.py"""

import os
import torch
from glob import glob
import matplotlib.pyplot as plt

from monai.transforms import (
    Compose,
    Spacingd,
    LoadImaged,
    EnsureTyped,
    ConcatItemsd,
    Orientationd,
    DivisiblePadD,
    CenterSpatialCropd,
    EnsureChannelFirstd,
    ScaleIntensityRangePercentilesd)

from monai.data import Dataset, DataLoader
from torch.utils.data import Subset, random_split

def read_paths_pair(root_dir: str):
    
    void_mri_files = sorted(glob(os.path.join(root_dir, "**", "*t1n-voided.nii.gz"), recursive=True))
    target_mri_files = sorted(glob(os.path.join(root_dir, "**", "*t1n.nii.gz"), recursive=True))
    healthy_mask_files = sorted(glob(os.path.join(root_dir, "**", "*mask-healthy.nii.gz"), recursive=True))
    unhealthy_mask_files = sorted(glob(os.path.join(root_dir, "**", "*mask-unhealthy.nii.gz"), recursive=True))
    mask_files = sorted(glob(os.path.join(root_dir, "**", "*mask.nii.gz"), recursive=True))

    return [{"mri": void, "mask": mask, "label": target, "healthy": healthy, "unhealthy": unhealthy} \
            for void, mask, target, healthy, unhealthy in \
            zip(void_mri_files, mask_files, target_mri_files, healthy_mask_files, unhealthy_mask_files)]

# MONAI Transforms
paired_transforms = Compose([
    LoadImaged(keys=["mri", "mask", "label"]),
    EnsureChannelFirstd(keys=["mri", "mask", "label"]),
    Orientationd(keys=["mri", "mask", "label"], axcodes="RAS"),
    Spacingd(
        keys=["mri", "mask", "label"],
        pixdim=(3, 3, 3),
        mode=("bilinear", "bilinear", "nearest")
    ),
    CenterSpatialCropd(keys=["mri", "mask", "label"], roi_size=(128, 128, 80)),
    DivisiblePadD(keys=["mri", "mask", "label"], k=16, mode="constant", constant_values=0),
    ScaleIntensityRangePercentilesd(
        keys=["mri", "label"],
        lower=0,
        upper=99.5,
        b_min=-1.0,
        b_max=1.0
    ),
    ConcatItemsd(keys=["mri", "mask"], name="input", dim=0),
    EnsureTyped(keys=["input", "label"]),
])

def get_dataloader(input_dir, batch_size=1):
    
    data_files = read_paths_pair(input_dir)
    paired_dataset = Dataset(data=data_files, transform=paired_transforms)
    subset = Subset(paired_dataset, indices = list(range(200)))
    train_dataset, validation_dataset = random_split(subset, [0.9, 0.1])
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    return train_loader, validation_loader