import numpy as np
import csv
import nibabel as nib
import skimage.transform

import matplotlib as mptlib
mptlib.use('TkAgg')
import matplotlib.pyplot as plt

import math
from scipy.signal import convolve2d

from preprocess import window

npx = 512
hu_lb = -50
hu_ub = 200

def resize_to_nn(img,transpose=True):
    if npx==img.shape[1]:
        expected = img
    else:
        expected = skimage.transform.resize(img,
            (npx,npx,img.shape[2]),
            order=0,
            mode='constant',
            preserve_range=True)
    if transpose:
        expected = expected.transpose(2,1,0)
    return expected

def reorient(imgloc, segloc=None):
    imagedata   = nib.load(imgloc)
    orig_affine = imagedata.affine
    orig_header = imagedata.header
    imagedata   = nib.as_closest_canonical(imagedata)
    img_affine  = imagedata.affine
    numpyimage  = imagedata.get_data().astype(np.int16)
    numpyseg    = None
    
    if segloc is not None:
        segdata     = nib.load(segloc)
        old_affine  = segdata.affine
        segdata     = nib.as_closest_canonical(segdata)
        seg_affine  = segdata.affine
        if not np.allclose(seg_affine, img_affine):
            segcopy = nib.load(segloc).get_data()
            copy_header = orig_header.copy()
            segdata = nib.nifti1.Nifti1Image(segcopy, orig_affine, header=copy_header)
            segdata = nib.as_closest_canonical(segdata)
            seg_affine = segdata.affine
        numpyseg = segdata.get_data().astype(np.uint8)

    return numpyimage, orig_header, numpyseg

dbfile_mda = '/rsrch1/ip/dtfuentes/github/RandomForestHCCResponse/datalocation/trainingdata.csv'
rootloc_mda = '/rsrch1/ip/dtfuentes/github/RandomForestHCCResponse'
def get_imgs(dbfile  = '/rsrch1/ip/jacctor/livermask/trainingdata_small.csv', rootloc = '/rsrch1/ip/jacctor/LiTS/LiTS'):

    nscans  = 0
    nvalid  = 0
    nslices = 0

    with open(dbfile, 'r') as csvfile:
        myreader=csv.DictReader(csvfile, delimiter='\t')

        for row in myreader:
            imageloc = '%s/%s' % (rootloc, row['image'])
            truthloc = '%s/%s' % (rootloc, row['label'])
            print(imageloc, truthloc)
            nscans += 1
            try:
                npimg, header, npseg = reorient(imageloc, segloc=truthloc)
                nslices += header['dim'][3]
                nvalid += 1
            except nib.filebasedimages.ImageFileError:
                print("could not read file")


    print('done precomputing size:  ', nslices, ' slices, from ', nvalid, ' scans out of ', nscans, ' scans.')
    imgs = np.empty((nslices,npx,npx))
    segs = np.empty((nslices,npx,npx))

    sidx = 0

    with open(dbfile, 'r') as csvfile:
        myreader = csv.DictReader(csvfile, delimiter='\t')
        for row in myreader:
            imageloc = '%s/%s' % (rootloc, row['image'])
            truthloc = '%s/%s' % (rootloc, row['label'])

            print(imageloc, truthloc)

            try:
                npimg, header, npseg = reorient(imageloc, segloc=truthloc)

                npimg = resize_to_nn(npimg, transpose=True).astype(np.int16)
                npseg = resize_to_nn(npseg, transpose=True).astype(np.uint8)

                sss = header['dim'][3]
                imgs[sidx:sidx+sss,...] = npimg
                segs[sidx:sidx+sss,...] = npseg
                sidx += sss
            except nib.filebasedimages.ImageFileError:
                print("ignoring the file I could't read earlier")

    return imgs, segs

def get_noise_2d(data, k=7):
    ker = np.ones((k,k))/(k**2) 
    mean = convolve2d(data,                   ker, mode='same', boundary='fill', fillvalue=0)
    var  = convolve2d(np.square(data - mean), ker, mode='same', boundary='fill', fillvalue=0)    
    return mean, np.sqrt(var)

# performs slicewise
# checking noise over 3d needs to deal with anisotropic voxels
def get_noise_3d(data3d, k=3):
    stdev = np.zeros_like(data3d)
    mean  = np.zeros_like(data3d)
    nslices = data3d.shape[0]
    for s in range(nslices):
        mean[s], stdev[s] = get_noise_2d(data3d[s,...], k=k)
    return mean, stdev

def show_histogram(data, idxlist=None, b=100, r=(-990,990)):
    if idxlist is not None:
        for idx in idxlist:
            to_show = data[idx].flatten()
            plt.hist(to_show, bins=b, range=r)
            plt.show()
            plt.close()
    else:
        to_show = data.flatten()
        plt.hist(to_show, bins=b, range=r)
        plt.show()
        plt.close()

def show_histogram_2D(datax, datay, idxlist=None, b=(100,100), r=[[-990,990],[0,60]]):
    if idxlist is not None:
        for idx in idxlist:
            to_show_x = datax[idx].flatten()
            to_show_y = datay[idx].flatten()
            plt.hist2d(to_show_x, to_show_y, bins=b, range=r)
            plt.show()
            plt.close()
    else:
        plt.hist2d(datax.flatten(), datay.flatten(), bins=b, range=r)
        plt.show()
        plt.close()


def process_std(data, idxlist=None, b=100, r=(-990,990)):
    mean, stdev = get_noise_3d(data, k=7)
    show_histogram(stdev, idxlist=idxlist, b=31, r=(0.1,59.9))
    return mean, stdev

def plot_histogram(data, idxlist=None, b=100, r=(-990,990), do_stdev=True):

    print(data.shape)
    show_histogram(data, idxlist=idxlist,  b=b, r=r)
    mean, stdev = process_std(data, idxlist=idxlist, b=b, r=r)
    show_histogram_2D(data, stdev,  idxlist=idxlist, b=(250,61), r=[[hu_lb+1, hu_ub-1],[0.1,59.9]])
    return mean, stdev



imgs, segs = get_imgs(dbfile=dbfile_mda,rootloc=rootloc_mda)
liver_idx = (segs > 0) * (segs < 5)
tumor_idx = (segs >= 2) * (segs <= 3) 
#only_liver_idx = liver_idx * (1.0 - tumor_idx)
all_idx = np.ones_like(segs, dtype=bool)

#ilist = [all_idx, liver_idx, tumor_idx, only_liver_idx.astype(bool)]
ilist = [all_idx, liver_idx, tumor_idx]
plot_histogram(imgs, idxlist=ilist,r=(hu_lb,hu_ub))

