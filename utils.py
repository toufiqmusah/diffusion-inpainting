"""utils.py"""

import os
import wandb
import matplotlib.pyplot as plt

from config import WEIGHT_DIR, LOG_FILE, GENERATED_DIR

def save_comparison(real_img, fake_img, input_img, epoch):
    os.makedirs(GENERATED_DIR, exist_ok=True)
    real_sample = real_img[0]
    fake_sample = fake_img[0]
    input_sample = input_img[0]

    print(f'Real: {real_sample.shape},  Fake: {fake_sample.shape}, Input: {input_sample.shape}')
    real_slice = real_sample[..., real_sample.shape[-1]//2]
    fake_slice = fake_sample[..., real_sample.shape[-1]//2]
    input_slice = input_sample[0,..., real_sample.shape[-1]//2]

    # to numpy and normalize
    def prepare_slice(slice_tensor):
        slice_np = slice_tensor.cpu().detach().numpy().squeeze()
        slice_np = (slice_np - slice_np.min()) / (slice_np.max() - slice_np.min())
        return slice_np

    real_np = prepare_slice(real_slice)
    fake_np = prepare_slice(fake_slice)
    input_np = prepare_slice(input_slice)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1.imshow(input_np, cmap='gray')
    ax1.set_title('Input Image')
    ax1.axis('off')
    ax2.imshow(fake_np, cmap='gray')
    ax2.set_title('Generated Image')
    ax2.axis('off')
    ax3.imshow(real_np, cmap='gray')
    ax3.set_title('Real Image')
    ax3.axis('off')

    plt.suptitle(f'Epoch {epoch}')
    plt.savefig(f'{GENERATED_DIR}/comparison_epoch_{epoch}.png', bbox_inches='tight', dpi=224)
    wandb.log({"Comparison": wandb.Image(plt.gcf(), caption=f"Epoch {epoch}")})
    plt.close()
