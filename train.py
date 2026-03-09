import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

import matplotlib.pyplot as plt
import datetime
import random
import string
import wandb
from tqdm import tqdm

# Import our own files
from datasets.celeba import getCelebADataloaders
from models.vae import VAE

use_cuda_if_avail = True
if use_cuda_if_avail and torch.cuda.is_available():
    device = "cuda"
    print("Using GPU")
else:
    device = "cpu"

config = {
    "bs":256,   # batch size
    "lr":0.0005, # learning rate
    "l2reg":0.0000001, # weight decay
    "max_epoch":200,
    "blocks": [32, 64, 128, 256, 512],
    "layers_per_scale": 2,
    "beta": 2,
    "image_size":64,
    "bottle":512
}


def main():

  # Get dataloaders
  train_loader, val_loader = getCelebADataloaders(config)

  # Build model
  model = VAE(block_dims=config["blocks"], layers_per_scale=config["layers_per_scale"], image_width=config["image_size"], bottle=config["bottle"])
  print(model)

  torch.compile(model)


  # Start model training
  train(model, train_loader, val_loader)


def computeBetaVAELoss(input, output, means, stdev, beta=1):
  sse_per_element = F.mse_loss(output, input, reduction="none")
  sse_per_sample = sse_per_element.reshape(sse_per_element.size(0), -1).sum(dim=1)
  sse_loss = sse_per_sample.mean()
  
  kl_per_sample = torch.sum(stdev**2/2 + means**2/2 - torch.log(stdev) - 0.5, dim=1)
  kl_loss = kl_per_sample.mean()
  loss = sse_loss + beta * kl_loss
  return loss, sse_loss, kl_loss


def train(model, train_loader, val_loader):

  # Log our exact model architecture string
  config["arch"] = str(model)
  run_name = generateRunName()

  # Startup wandb logging
  wandb.login()
  wandb.init(project="CelebA CS499 A5", name=run_name, config=config)

  # Move model to the GPU
  model.to(device)

  # Set up optimizer and our learning rate schedulers
  optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=config["l2reg"])
  warmup_epochs = config["max_epoch"]//10
  linear = LinearLR(optimizer, start_factor=0.25, total_iters=warmup_epochs)
  cosine = CosineAnnealingLR(optimizer, T_max = config["max_epoch"]-warmup_epochs)
  scheduler = SequentialLR(optimizer, schedulers=[linear, cosine], milestones=[warmup_epochs])

  # Main training loop with progress bar
  iteration = 0
  pbar = tqdm(total=config["max_epoch"]*len(train_loader), desc="Training Iterations", unit="batch")
  for epoch in range(config["max_epoch"]):
    model.train()

    # Log LR
    wandb.log({"LR/lr": scheduler.get_last_lr()[0]}, step=iteration)

    for x, _ in train_loader:
      x = x.to(device)

      out, means, stdev = model(x)
      loss, recon, kl = computeBetaVAELoss(x, out, means, stdev, beta=config["beta"])

      loss.backward()
      optimizer.step()
      optimizer.zero_grad()


      wandb.log({"Loss/train": loss.item(), "Loss/train_KL": kl, "Loss/train_recon": recon}, step=iteration)
      pbar.update(1)
      iteration+=1


    if epoch % 5 == 0:
      # Plot reconstructions
      f = generateReconstructionPlot(model, val_loader)
      wandb.log({"Viz/reconstruct":f}, step=iteration)
      plt.close(f)

      # Plot samples
      f = generateSamplePlot(model, val_loader)
      wandb.log({"Viz/samples":f}, step=iteration)
      plt.close(f)

    # Adjust LR
    scheduler.step()

  wandb.finish()
  pbar.close()




def generateRunName():
  random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
  now = datetime.datetime.now()
  run_name = ""+random_string+"_CelebA"
  return run_name


def generateReconstructionPlot(model, test_loader):
  model.eval()
  x,y = next(iter(test_loader))

  out, _, _ = model(x.to(device))
  out = out.detach()
  out = torch.clamp(out*0.5+0.5, 0,1)

  x = torch.clamp(x*0.5+0.5, 0, 1)

  num = min(20, x.shape[0])
  f, axs = plt.subplots(2, num, figsize=(2*num,4))
  for i in range(0,num):

    axs[0,i].imshow(x[i,:,:,:].squeeze().permute(1,2,0).cpu())
    axs[0,i].get_xaxis().set_visible(False)
    axs[0,i].get_yaxis().set_visible(False)

    axs[1,i].imshow(out[i,:,:,:].squeeze().permute(1,2,0).cpu())
    axs[1,i].get_xaxis().set_visible(False)
    axs[1,i].get_yaxis().set_visible(False)

  f.tight_layout()
  return f

def generateSamplePlot(model, test_loader):
  model.eval()

  num = 100  
  samples = model.bottleneck.generateRandomSamples(num, device)
  out = model.decoder(samples).detach()
  out = torch.clamp(out*0.5+0.5, 0,1)
  
  row = 5
  col = num // 5
  f, axs = plt.subplots(row, col, figsize=(2*col,2*row))
  for i in range(0,num):

    axs[i//col, i%col].imshow(out[i,:,:,:].squeeze().permute(1,2,0).cpu())
    axs[i//col, i%col].get_xaxis().set_visible(False)
    axs[i//col, i%col].get_yaxis().set_visible(False)
  
  f.tight_layout()
  return f


main()
