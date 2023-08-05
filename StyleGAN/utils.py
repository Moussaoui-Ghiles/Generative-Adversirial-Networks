# import some packages 


import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from scipy.stats import truncnorm 
import numpy as np 
import torchvision 
from torchvision.utils import make_grid
import matplotlib.pyplot as plt 

# make a show function that desplays the pics

def show_tensor_images(image_tensor, num_images = 16, size = (3, 64, 64), nrow = 3):
    '''
    Function for visualizing images: Given a tensor of images, number of images, 
    size per image, and images per row, plots and prints the images in an unifrm grid.
    '''
    
    image_tensor = (image_tensor + 1)/2
    image_unflat = image_tensor.detach().cpu().clamp_(0, 1)
    image_grid = make_grid(image_unflat[: num_images], nrow = nrow, padding = 0)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.axis("off")
    plt.show()
    
# The first component to implement is the Truncation Trick


def get_truncated_noise(n_samples, z_dim, truncation):
    '''
    Function for creating truncated noise vectors: Given the dimensions (n_samples, z_dim)
    and truncation value, creates a tensor of that shape filled with random
    numbers from the truncated normal distribution.
    Parameters:
        n_samples: the number of samples to generate, a scalar
        z_dim: the dimension of the noise vector, a scalar
        truncation: the truncation value, a non-negative scalar
        
    '''
    truncated_noise = truncnorm.rvs(-1*truncation, truncation , size = (n_samples, z_dim))
    return torch.Tensor(truncated_noise)

    