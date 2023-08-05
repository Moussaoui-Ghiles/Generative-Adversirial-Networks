# Here we implement the deffiret component of the generator
# first is the mapping network "W", which takes the noise vector z and maps it to intermediate noise vector w

import torch
import torch.nn as nn 

import torch.nn.functional as F
from utils import get_truncated_noise

class MappingLayers(nn.Module):
    
    '''
    Mapping Layers Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        hidden_dim: the inner dimension, a scalar
        w_dim: the dimension of the intermediate noise vector, a scalar
    '''
    def __init__(self, z_dim, hidden_dim, w_dim):
        
        super().__init__()
        self.mapping = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, w_dim),
            
        )
    def forward(self, noise):
        '''
        Function for completing a forward pass of MappingLayers:
        Given an initial noise tensor, returns the intermediate noise tensor.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''
        return self.mapping(noise)
    
    def get_mapping(self):
        return self.mapping 
    
# Random Noise Injection:
class InjectNoise(nn.Module):
    '''
    Inject Noise Class
    Values:
        channels: the number of channels the image has, a scalar
    '''
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(# You use nn.Parameter so that these weights can be optimized
                                   torch.randn(channels)[None, :, None, None]
                                   )
    def forward(self, image):
        
         '''
        Function for completing a forward pass of InjectNoise: Given an image,
        returns the image with random noise added.
        Parameters:
            image: the feature map of shape (n_samples, channels, width, height)
        '''
        # the appropriate shape for the noise!
        
         noise_shape = (image.shape[0], 1, image.shape[2], image.shape[3])
        
         noise = torch.randn(noise_shape, device = image.device) # Create the Random noise 
         return image + self.weight * noise # Added to the image after multiplying by the weight for each channel
    def get_weight(self):
        return self.weight
    def get_self(self):
        return self
# Adaptive instance Normalization (AdaIN), the next component

   
class AdaIN(nn.Module):
    '''
    AdaIN Class
    Values:
        channels: the number of channels the image has, a scalar
        w_dim: the dimension of the intermediate noise vector, a scalar
    '''
    def __init__(self, channels, w_dim):
        super().__init__()
        
        # Normalize the input per-dimension
        self.instance_norm = nn.InstanceNorm2d(channels)
        self.style_scale_transform = nn.Linear(w_dim, channels)
        self.style_shift_transform = nn.Linear(w_dim, channels)
        
        
    def forward(self, image, w):
        '''
        Function for completing a forward pass of AdaIN: Given an image and intermediate noise vector w,
        returns the normalized image that has been scaled and shifted by the style.
        Parameters:
            image: the feature map of shape (n_samples, channels, width, height)
            w: the intermediate noise vector
        '''
        normalized_image = self.instance_norm(image)
        style_scale = self.style_scale_transform(w)[:, :, None, None]
        style_shift = self.style_shift_transform(w)[:, :, None, None]
        
        # Calculate the transformed image
        transformed_image = style_scale * normalized_image + style_shift
        
        return transformed_image
    
    def get_style_scale_transform(self):
        return self.style_scale_transform
    def get_style_shift_transform(self):
        return self.style_shift_transform
    def get_self(self):
        return self
    
# Progressive Growing in StyleGAN

class MicroStyleGANGeneratorBlock(nn.Module):
    '''
    Micro StyleGAN Generator Block Class
    Values:
        in_chan: the number of channels in the input, a scalar
        out_chan: the number of channels wanted in the output, a scalar
        w_dim: the dimension of the intermediate noise vector, a scalar
        kernel_size: the size of the convolving kernel
        starting_size: the size of the starting image
    '''
    def __init__(self, in_chan, out_chan, w_dim, kernel_size, starting_size, use_upsample = True ):
        
        super().__init__()
        self.use_upsample = use_upsample
        
        if self.use_upsample:
            self.upsample = nn.Upsample((starting_size), mode = "bilinear")
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size, padding=1) #Padding is used to maintain the image size 
        self.inject_noise = InjectNoise(out_chan)
        self.adin = AdaIN(out_chan, w_dim)
        self.activation = nn.LeakyReLU(0.2)
        
        
    def forward(self, x, w):
        '''
        Function for completing a forward pass of MicroStyleGANGeneratorBlock: Given an x and w,
        computes a StyleGAN generator block.
        Parameters:
            x: the input into the generator, feature map of shape (n_samples, channels, width, height)
            w: the intermediate noise vector
        '''
        if self.use_upsample:
            x = self.upsample(x)
        x = self.conv(x)
        x = self.inject_noise(x)
        x = self.activation(x)
        x = self.adin(x, w)
        return x
    def get_self(self):
        return self;
            
    
    
    
# Mini Stylegan generator 

class MicroStyleGANGenerator(nn.Module):
    '''
    Micro StyleGAN Generator Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        map_hidden_dim: the mapping inner dimension, a scalar
        w_dim: the dimension of the intermediate noise vector, a scalar
        in_chan: the dimension of the constant input, usually w_dim, a scalar
        out_chan: the number of channels wanted in the output, a scalar
        kernel_size: the size of the convolving kernel
        hidden_chan: the inner dimension, a scalar
    '''
    def __init__(self,
                 z_dim,
                 map_hidden_dim,
                 w_dim,
                 in_chan,
                 out_chan,
                 kernel_size,
                 hidden_chan):
        super().__init__()
        self.map = MappingLayers(z_dim, map_hidden_dim, w_dim)
        self.starting_constant = nn.Parameter(torch.randn(1, in_chan, 4, 4))
        self.block0 = MicroStyleGANGeneratorBlock(in_chan, hidden_chan, w_dim, kernel_size, 4, use_upsample = False)
        self.block1 = MicroStyleGANGeneratorBlock(hidden_chan, hidden_chan, w_dim, kernel_size, 8)
        self.block2 = MicroStyleGANGeneratorBlock(hidden_chan, hidden_chan, w_dim, kernel_size, 16)
        # transform to an image 
        self.block1_to_image = nn.Conv2d(hidden_chan, out_chan, kernel_size=1)
        self.block2_to_image = nn.Conv2d(hidden_chan, out_chan, kernel_size=1)
        self.alpha = 0.2
        
    def upsample_to_match_size(self, smaller_image, bigger_image):
        '''
        Function for upsampling an image to the size of another: Given a two images (smaller and bigger),
        upsamples the first to have the same dimensions as the second.
        Parameters:
            smaller_image: the smaller image to upsample
            bigger_image: the bigger image whose dimensions will be upsampled to
        '''
        return F.interpolate(smaller_image, size = bigger_image.shape[-2:], mode = "bilinear")
    
    def forward(self, noise, return_intermediate=False):
        '''
        Function for completing a forward pass of MicroStyleGANGenerator: Given noise,
        computes a StyleGAN iteration.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
            return_intermediate: a boolean, true to return the images as well (for testing) and false otherwise
        '''
        x = self.starting_constant
        w = self.map(noise)
        x = self.block0(x, w)
        x_small = self.block1(x, w) #First generator run output
        x_small_image = self.block1_to_image(x_small)
        x_big = self.block2(x_small, w) #Second generator run output
        x_big_image = self.block2_to_image(x_big)
        x_small_upsample = self.upsample_to_match_size(x_small_image, x_big_image) # Upsample first generator run outputto same size as second generator run output
        
        interpolation = self.alpha * (x_big_image) + (1 - self.alpha) * (x_small_upsample)
        
        
        if return_intermediate:
            return interpolation, x_small_upsample, x-x_big_image
        return interpolation
    
    def get_self(self):
        return self;
        
        
    
        
z_dim = 128
out_chan = 3
truncation = 0.7

mu_stylegan = MicroStyleGANGenerator(
    z_dim=z_dim,
    map_hidden_dim=1024,
    w_dim=496,
    in_chan=512,
    out_chan=out_chan,
    kernel_size=3,
    hidden_chan=256
)

test_samples = 10
test_result = mu_stylegan(get_truncated_noise(test_samples, z_dim, truncation))

# Check if the block works
assert tuple(test_result.shape) == (test_samples, out_chan, 16, 16)

# Check that the interpolation is correct
