"""train.py"""
import time
import torch
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from generative.inferers import DiffusionInferer
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler

from utils import save_comparison

scaler = GradScaler()
total_start = time.time()

def train_diffusion(model, train_loader, val_loader, device, n_epochs=10, val_interval=2):

    epoch_loss_list = []
    val_epoch_loss_list = []

    scheduler = DDIMScheduler(num_train_timesteps=1000,  
                          schedule="sigmoid_beta",
                          beta_start=1e-4,           
                          beta_end=2e-2,               
                          sig_range=8)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=2.5e-5) 
    inferer = DiffusionInferer(scheduler)

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0

        for step, data in enumerate(train_loader):
            input = data["input"].to(device)
            real = data["label"].to(device)  
            optimizer.zero_grad(set_to_none=True)
            timesteps = torch.randint(0, 1000, (len(input),)).to(device) 

            with autocast(enabled=True):
                noise = torch.randn_like(real).to(device)
                noisy_real = scheduler.add_noise(
                    original_samples=real, noise=noise, timesteps=timesteps
                )  # we only add noise to the realmentation mask

                combined = torch.cat(
                    (input, noisy_real), dim=1
                )  # we concatenate the brain MR image with the noisy realmenatation mask, to condition the generation process
                prediction = model(x=combined, timesteps=timesteps)
                # Get model prediction

                loss = F.mse_loss(prediction.float(), noise.float())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        epoch_loss_list.append(epoch_loss / (step + 1))
        if (epoch) % val_interval == 0:
            model.eval()
            val_epoch_loss = 0
            for step, data_val in enumerate(val_loader):
                input = data_val["input"].to(device)
                real = data_val["label"].to(device)  
                timesteps = torch.randint(0, 1000, (len(input),)).to(device)
                with torch.no_grad():
                    with autocast(enabled=True):
                        noise = torch.randn_like(real).to(device)
                        noisy_real = scheduler.add_noise(original_samples=real, noise=noise, timesteps=timesteps)
                        combined = torch.cat((input, noisy_real), dim=1)
                        prediction = model(x=combined, timesteps=timesteps)
                        val_loss = F.mse_loss(prediction.float(), noise.float())
                val_epoch_loss += val_loss.item()
            print("Epoch", epoch, "Validation loss", val_epoch_loss / (step + 1))
            val_epoch_loss_list.append(val_epoch_loss / (step + 1))
        
        if (epoch + 1) % 1 == 0:
            pass
            #save_comparison(real_img=real, fake_img=prediction, input_img=input, epoch=epoch + 1)


    torch.save(model.state_dict(), "./DiffusionModel.pt")
    total_time = time.time() - total_start
    print(f"train diffusion completed, total time: {total_time}.")
    plt.style.use("seaborn-bright")
    plt.title("Learning Curves Diffusion Model", fontsize=20)
    plt.plot(np.linspace(1, n_epochs, n_epochs), epoch_loss_list, color="C0", linewidth=2.0, label="Train")
    plt.plot(
        np.linspace(val_interval, n_epochs, int(n_epochs / val_interval)),
        val_epoch_loss_list,
        color="C1",
        linewidth=2.0,
        label="Validation",
    )
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.xlabel("Epochs", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.legend(prop={"size": 14})
    plt.show()