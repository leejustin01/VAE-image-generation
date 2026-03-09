from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torch

def getCelebADataloaders(config):

  # Set up dataset and data loaders
  test_transform=transforms.Compose([
      transforms.ToTensor(),
      transforms.Resize((config["image_size"],config["image_size"])),
      transforms.Normalize((0.5,), (0.5,))
      ])

  
  train_transform=transforms.Compose([
      transforms.ToTensor(),
      transforms.RandomHorizontalFlip(),
      transforms.Resize((config["image_size"],config["image_size"])),
      transforms.Normalize((0.5,), (0.5,))
      ])
 

  train_set = datasets.CelebA(root='./datasets/data', split="valid",download=True, transform=train_transform)
  val_set = datasets.CelebA(root='./datasets/data', split="test",download=True, transform=test_transform)

  train_loader = DataLoader(train_set, shuffle=True, batch_size=config["bs"], num_workers=8, drop_last=True, pin_memory=True)
  val_loader = DataLoader(val_set, shuffle=False, batch_size=20, num_workers=8, drop_last=True, pin_memory=True)

  return train_loader, val_loader