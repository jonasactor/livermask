import numpy as np
import time

# assumes 3x3 conv kernels
def spectral_norm_conv(nx, ny, k):

    import numpy as np
    import scipy as sp
    from scipy import linalg, sparse
    from scipy.sparse import linalg

    c_in  = k.shape[2]
    c_out = k.shape[3]
    A = sp.sparse.csr_matrix((nx*ny*c_out, nx*ny*c_in))

    for c in range(c_in):
        for d in range(c_out):

            K0 = sp.sparse.diags(k[0,:,c,d], [-1,0,1], shape=(nx,nx))
            K1 = sp.sparse.diags(k[1,:,c,d], [-1,0,1], shape=(nx,nx))
            K2 = sp.sparse.diags(k[2,:,c,d], [-1,0,1], shape=(nx,nx))

            J0 = sp.sparse.kron(sp.sparse.diags([1], [-1], shape=(ny,ny)), K0)
            J1 = sp.sparse.kron(sp.sparse.diags([1], [0],  shape=(ny,ny)), K1)
            J2 = sp.sparse.kron(sp.sparse.diags([1], [1],  shape=(ny,ny)), K2)

            Acd = J0 + J1 + J2

            A[d*nx*ny:(d+1)*nx*ny, c*nx*ny:(c+1)*nx*ny] = Acd

    AtA = A.T * A
    nrm2, _ = sp.sparse.linalg.eigs(AtA, k=1)
    nrm = np.sqrt(np.real(nrm2))
    print(nrm)
    return nrm


for jjj in range(1,6):
    nx = 2**jjj
    ny = 2**jjj
    c_in  = 16
    c_out = 16

    k = np.random.randn(3,3,c_in,c_out)

    ttt0 = time.time()
    nnn = spectral_norm_conv(nx, ny, k)
    ttt1 = time.time()
    print(jjj,' time elapsed:\t', ttt1-ttt0)

