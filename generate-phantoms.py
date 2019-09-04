import numpy as np
import nibabel as nib
from scipy import ndimage

nx = 512
ny = 512
nz = 8
x = np.linspace(0, nx-1, nx)
y = np.linspace(0, ny-1, ny)
xv, yv = np.meshgrid(x,y)

for k in range(30):
    circle_val = np.zeros((nx,ny))
    for kk in range(10):
        cx = np.random.randint(106,406)
        cy = np.random.randint(106,406)
        r  = np.random.randint(10,50)

        circle_fun = lambda i,j:   (i-cx)**2 + (j-cy)**2  
        for i in range(nx):
            for j in range(ny):
                if circle_fun(xv[i,j], yv[i,j]) < r**2 :
                    circle_val[i,j] += 1.0
    circle_val   = np.minimum(circle_val, np.ones_like(circle_val)) 
    circle       = np.tile(circle_val, nz).reshape((nx,nz,ny)).transpose((0,2,1))
    circle_nifti = nib.Nifti1Image(circle, affine=np.eye(4))
    nib.save(circle_nifti, "./circles/circle-seg-"+str(k)+".nii.gz")

    noise        = np.random.normal(loc=0.0, scale=0.1, size=np.shape(circle))
    background   = np.tile(4.0*(xv + yv)/(nx+ny),nz).reshape((nx,nz,ny)).transpose((0,2,1))
    circle       = 256*(circle + noise + background) / np.max(circle+noise+background)
    circle_nifti = nib.Nifti1Image(circle, affine=np.eye(4))
    nib.save(circle_nifti, "./circles/circle-vol-"+str(k)+".nii.gz")

    print("\tDone ", k+1, " circles.")
