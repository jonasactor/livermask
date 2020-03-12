import numpy as np
import time

# assumes 3x3 conv kernels

def spectral_norm_conv_iterative(nx, ny, k):

    import numpy as np
    import scipy as sp
    from scipy import signal

    c_in  = k.shape[2]
    c_out = k.shape[3]
    x = np.random.randn(ny,nx,c_in)
    n=0
    maxiter = 10000
    tol=0.001
    err=np.inf

    ker  = k
    kerT = k[::-1,::-1,...]
#    kerT = np.fliplr(np.flipud(ker))
    kerT = np.swapaxes(kerT, 2, 3)

    while err > tol:

        x_prev = x
        
        y = np.zeros((ny,nx,c_out))
        z = np.zeros((ny,nx,c_in))

        for ci in range(c_in):
            for co in range(c_out):
                y[...,co] += sp.signal.convolve2d(x[...,ci], ker[...,ci,co],  mode='same', boundary='fill')

        for co in range(c_out):
           for ci in range(c_in):
                z[...,ci] += sp.signal.convolve2d(y[...,co], kerT[...,co,ci], mode='same', boundary='fill')

        x = z / np.linalg.norm(z)

        err = np.linalg.norm(x_prev - x)
        n += 1
        if n > maxiter:
            break

    # compute rayleigh quotient = norm
    Ax = np.zeros((ny,nx,c_out))
    for ci in range(c_in):
        for co in range(c_out):
            Ax[...,co] += sp.signal.convolve2d(x[...,ci], ker[...,ci,co], mode='same', boundary='fill')
    return np.linalg.norm(Ax) / np.linalg.norm(x), n, err

def spectral_norm_conv_direct(nx, ny, k):

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
    return nrm


for jjj in range(1,8):
    nx = 2**jjj
    ny = 2**jjj
    c_in  = 16 
    c_out = 16 

    k = np.random.randn(3,3,c_in,c_out)

#    ttt0 = time.time()
#    ddd = spectral_norm_conv_direct(ny, nx, k)
    ddd=0
    ttt1 = time.time()
    iii, nit, er = spectral_norm_conv_iterative(ny, nx, k)
    ttt2 = time.time()
    print(jjj)
    print('\t matrix size:          \t', nx,'.',ny,'.',c_out,' x ',nx,'.',ny,'.',c_in,' = ',nx*ny*c_out,' x ',nx*ny*c_in)
    print('\t kernel size:          \t', k.shape )
 #   print('\t time elapsed:    \t', ttt1-ttt0, ttt2-ttt1)
 #   print('\t calculated norms:\t', ddd, iii)
    print('\t time elapsed:         \t', ttt2-ttt1)
    print('\t calculated norms:     \t', iii)
    print('\t number iterations:    \t', nit)
    print('\t error in eigenvectors:\t', er)
