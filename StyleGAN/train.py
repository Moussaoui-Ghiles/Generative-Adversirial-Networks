import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from scipy.stats import truncnorm 
import numpy as np 
import torchvision 
from torchvision.utils import make_grid
import matplotlib.pyplot as plt 
plt.rcParams["figure.figsize"] = [15, 15]
from utils import get_truncated_noise, show_tensor_images
from model import mu_stylegan
z_dim = 128
out_chan = 3
truncation = 0.7

viz_sample = 10 
# The noise is exaggerated for the visual effect

viz_noise = get_truncated_noise(viz_sample, z_dim, truncation) * 10

mu_stylegan.eval()
images = []
for alpha in np.linspace(0, 1, num=5):
    mu_stylegan.alpha = alpha 
    viz_result, _, _ = mu_stylegan(
        viz_noise,
        return_intermediate=True
    )
    images +=[tensor for tensor in viz_result]
show_tensor_images(torch.stack(images), nrow = viz_sample, num_images = len(images))
mu_stylegan = mu_stylegan.train()