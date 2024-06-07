import numpy as np

import torch
from torchvision import transforms

def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1/25):
    alpha = np.random.uniform(*alpha_range)
    noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
    return alpha * data + beta * noise

# batch image random mask
def random_mask_batch_image(input_batch, mask_ratio): # input (batchsize, 128, 7, 7)
    batch_size = input_batch.shape[0]
    num_channels = input_batch.shape[1]
    patch_size = input_batch.shape[2]
    random_mask_spatial = torch.rand(batch_size, 1, patch_size, patch_size)
    random_mask_spatial = torch.where(random_mask_spatial > mask_ratio, torch.tensor(1.0), torch.tensor(0.0))
    masked_batch = input_batch * random_mask_spatial
    return masked_batch