import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm2d(nn.LayerNorm):
  def forward(self, x):
    x = x.permute(0, 2, 3, 1)
    x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    x = x.permute(0, 3, 1, 2)
    return x

class DownsizeBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    
    self.norm = LayerNorm2d(in_channels)
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
    self.gelu = nn.GELU()
  def forward(self, x):
    x = self.norm(x)
    x = self.conv(x)
    x = self.gelu(x)
    return x

class UpsizeBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.norm = LayerNorm2d(in_channels)
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    self.gelu = nn.GELU()
    

  def forward(self, x):
    x = F.interpolate(x, scale_factor=2, mode='bilinear')
    x = self.norm(x)
    x = self.conv(x)
    x = self.gelu(x)
    return x


class Bottleneck(nn.Module):
  def __init__(self, input_shape, bottle_dim=64):
    super().__init__()
    self.C = input_shape[0]
    self.W = input_shape[1]
    self.H = input_shape[2]
    self.bottle_dim = bottle_dim
    
    D = self.C * self.W * self.H
    self.norm = nn.LayerNorm(D)
    self.mean_linear = nn.Linear(D, bottle_dim)
    self.stdev_linear = nn.Linear(D, bottle_dim)
    self.softplus = nn.Softplus()
    self.return_linear = nn.Linear(bottle_dim, D)
    
  
  def sample(self, means, stds):
    noise = torch.randn_like(means)
    return means + stds * noise

  def generateRandomSamples(self, b, device):
    noise = torch.randn((b, self.bottle_dim)).to(device)
    out = self.return_linear(noise)
    out = torch.reshape(out, (b, self.C, self.W, self.H))
    return out

  def forward(self, x):
    flat = torch.flatten(x, start_dim=1)
    normalized = self.norm(flat)
    mean = self.mean_linear(normalized)
    stdev = self.stdev_linear(normalized)
    stdev = self.softplus(stdev)
    stdev = torch.clamp(stdev, min=0.0001, max=2)
    z = self.sample(mean, stdev)
    out = self.return_linear(z)
    out = torch.reshape(out, (x.size(0), self.C, self.W, self.H))
    return out, mean, stdev
   
class ResidualConvBlock(nn.Module):
  def __init__(self, d, layers, width=3):
    super().__init__()
      
    self.layers = nn.ModuleList(nn.Conv2d(d, d, kernel_size=width, stride=1, padding=(width//2)) for i in range(layers))
    self.gelu = nn.GELU()
  def forward(self, x):
    
    residual = x
    
    for i in range(len(self.layers)):
      if i > 0:
        x = self.gelu(x)
      x = self.layers[i](x)

    return residual + x

class VAE(nn.Module):

  def __init__(self, block_dims = [16, 32, 64, 128], layers_per_scale=2, image_width=64, bottle=512):
    super().__init__()
    
    encoder_layers = [nn.Conv2d(3, block_dims[0], kernel_size=3, stride=1, padding=1)]
    for i in range(len(block_dims)):
      encoder_layers.append(ResidualConvBlock(block_dims[i], layers_per_scale))
      if i != (len(block_dims) - 1):
        downsize_dim = block_dims[i+1]
      else:
        downsize_dim = block_dims[i]
      encoder_layers.append(DownsizeBlock(block_dims[i], downsize_dim))

    self.encoder = nn.Sequential(*encoder_layers)
      
    S = image_width // (2**len(block_dims))
    self.bottleneck = Bottleneck([block_dims[-1], S, S], bottle_dim=bottle)
    
    decoder_layers = []
    for j in range(len(block_dims)):
      i = len(block_dims) - j - 1  # fix indexing: Python zero-based
      decoder_layers.append(ResidualConvBlock(block_dims[i], layers_per_scale))
      if i != 0:
        upsize_dim = block_dims[i-1]
      else:
        upsize_dim = block_dims[i]
      decoder_layers.append(UpsizeBlock(block_dims[i], upsize_dim))

    decoder_layers.append(nn.Conv2d(upsize_dim, block_dims[0], kernel_size=3, padding=1))
    decoder_layers.append(nn.GELU())
    decoder_layers.append(nn.Conv2d(block_dims[0], 3, kernel_size=1))
    decoder_layers.append(nn.Tanh())

    self.decoder = nn.Sequential(*decoder_layers)
    
    # initialize parameters
    for module in self.modules():
      if isinstance(module, Bottleneck):
        nn.init.normal_(module.mean_linear.weight, mean=0.0, std=0.002)
        nn.init.normal_(module.stdev_linear.weight, mean=0.0, std=0.002)
        nn.init.zeros_(module.mean_linear.bias)
        nn.init.constant_(module.stdev_linear.bias, 0.541234)
      elif isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, mean=0, std=0.02)
        nn.init.zeros_(module.bias)
      elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
      

  def forward(self, x):
    x = self.encoder(x)
      
    x, mean, stdev = self.bottleneck(x)
    
    x = self.decoder(x)
    
      
    return x, mean, stdev
    

