import torch
import torch.nn as nn 
import torch.nn.functional as F 
from math import log2

factors = [1, 1, 1, 1, 1/2, 1/4, 1/8, 1/16, 1/32]

class WSConv2d(nn.Module):# (weited scaled convulutional layers)
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, gain = 2 ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (gain/ (in_channels * kernel_size ** 2)) ** 0.5
        self.bias = self.conv.bias 
        self.conv.bias = None 
        
        # iNITIALIZE CONV LAYER 
        nn.init_normal_(self.conv.weight)
        nn.init_zeros_(self.bias)
        
    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)
    
class PixelNorm(nn.Module):
    def __init__(self ):
        super().__init__()
        self.epsilon = 1e - 8 
        
        def forward(self, x):
            return x / torch.sqrt(torch.mean(x**2, dim = 1, keepdim = True) + self.epsilon)
        
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_pixelnorm = True):
        super().__init__()
        self.conv1 = WSConv2d(in_channels, out_channels)
        self.conv2 = WSConv2d(out_channels, out_channels)
        self.leaky = nn.LeakyReLU(0.2)
        self.pn = PixelNorm()
        self.use_pn = use_pixelnorm
        
    def forward(self, X):
        
        X = self.leaky(self.conv1(x))
        x = self.pn(x) if self.use_pn else x
        x = self.leaky(self.conv2(x))
        x = self.pn(x) if self.use_pn else x
        return x
        
class Generator(nn.Module):
    def __init__(self, z_dim, in_channels, img_channels):
        super().__init__()
        self.initial = nn.Sequential(
            PixelNorm(),
            nn.ConvTranspose2d(z_dim, in_channels, 4, 1, 0), #1x1 -> 4x4
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            PixelNorm(),
             
        ) 
class Discrominator(nn.Module):
    pass

    