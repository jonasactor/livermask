import numpy as np
from matplotlib import pyplot as plt

from makenoise import *

nz = 1
data = np.zeros((256,256,1))
data[:150,:80,:] = 0.5
data[60:,130:,:] = -0.5
data[0,0,0] = -1.0
data[1,0,0] = 1.0

for jj in range(nz):
    plt.imshow(data[:,:,jj])
    plt.show()

    corrupted = make_gaussian_noise(data,eps=16.)
    plt.imshow(corrupted[:,:,jj])
    plt.show()

    corrupted = make_uniform_noise(data,eps=16.)
    plt.imshow(corrupted[:,:,jj])
    plt.show()
    
    corrupted = make_sparse_gaussian_noise(data, eps=16, sparsity=0.0625)
    plt.imshow(corrupted[:,:,jj])
    plt.show()

    corrupted = make_sparse_uniform_noise(data,eps=16., sparsity=0.0625)
    plt.imshow(corrupted[:,:,jj])
    plt.show()




