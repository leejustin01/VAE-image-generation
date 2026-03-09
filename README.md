# Variational Autoencoder for Image Generation

An implementation of a Variational Autoencoder (VAE) for generating images by learning a compressed latent representation of the CelebA. The model learns to encode images into a latent space and generate new images by sampling from that distribution.

## Overview

This project trains a Variational Autoencoder to learn the underlying distribution of an image dataset. The encoder compresses images into a lower-dimensional latent representation, while the decoder reconstructs images from sampled latent vectors, enabling the generation of new images.

## Features

- Image generation using a Variational Autoencoder (VAE)  
- Encoder–decoder architecture with a learned latent space  
- Sampling from latent distributions to generate new images  
- Image reconstruction and generation from latent vectors  

## Example

```python
train_loader, val_loader = getCelebADataloaders(config)
model = VAE(block_dims=config["blocks"], layers_per_scale=config["layers_per_scale"], image_width=config["image_size"], bottle=config["bottle"])
torch.compile(model)
train(model, train_loader, val_loader)
```

Acknowledgement

This project was developed as part of a university assignment for **CS 435 - Applied Deep Learning** at Oregon State University.
