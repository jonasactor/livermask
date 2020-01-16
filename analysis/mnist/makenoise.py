import numpy as np
from matplotlib import pyplot as plt

# makes noise
# assumes input images are in range [-1,1]
# data : np array
# eps  : magnitude of noise



# noise follows N(0, 1/sqrt(3))
# so that 99% of noise falls in [-1,1]
# then truncated back to [-1,1]
def make_gaussian_noise(data, eps=0.01):
    noise      = np.random.normal(0, 1.0/np.sqrt(3.0), data.shape)
    norm_noise = np.linalg.norm(noise)
    corrupted  = data + (eps / norm_noise) * noise
    return np.clip(corrupted, -1.0, 1.0)

# noise follows Unif[-1,1]
def make_uniform_noise(data, eps=0.01):
    noise      = np.random.uniform(-1.0, 1.0, data.shape)
    norm_noise = np.linalg.norm(noise)
    corrupted  = data + (eps / norm_noise) * noise
    return np.clip(corrupted, -1.0, 1.0)

# noise follows N(0, 1/sqrt(3))
# then truncated as before,
# but on average only sparsity % of pixels are corrupted
def make_sparse_gaussian_noise(data, eps=0.01, sparsity=0.01):
    subset     = np.random.binomial(1, sparsity, size=data.shape)
    noise      = np.random.normal(0, 1.0/np.sqrt(3.0), data.shape)
    noise      = np.multiply(subset, noise)
    norm_noise = np.linalg.norm(noise)
    corrupted  = data + (eps / norm_noise) * noise
    return np.clip(corrupted, -1.0, 1.0)
    
# noise follows Unif[-1,1]
# but only sparsity % of pixels are corrupted
def make_sparse_uniform_noise(data, eps, sparsity):
    subset     = np.random.binomial(1, sparsity, size=data.shape)
    noise      = np.random.uniform(-1.0, 1.0, data.shape)
    noise      = np.multiply(subset, noise)
    norm_noise = np.linalg.norm(noise)
    corrupted  = data + (eps / norm_noise) * noise
    return np.clip(corrupted, -1.0, 1.0)


