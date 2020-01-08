import numpy as np
import csv
import nibabel as nib
import skimage.transform

import matplotlib as mptlib
mptlib.use('TkAgg')
import matplotlib.pyplot as plt


def resize_to_nn(img,transpose=True):
    expected = skimage.transform.resize(img,
            (256,256,img.shape[2]),
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

def get_imgs(dbfile  = '/rsrch1/ip/jacctor/livermask/trainingdata_one.csv', rootloc = '/rsrch1/ip/jacctor/LiTS/LiTS'):

    imgs = np.empty((0,256,256))
    segs = np.empty((0,256,256))

    with open(dbfile, 'r') as csvfile:
        myreader = csv.DictReader(csvfile, delimiter=',')
        for row in myreader:
            imageloc = '%s/%s' % (rootloc, row['image'])
            truthloc = '%s/%s' % (rootloc, row['label'])

            print(imageloc, truthloc)

            npimg, header, npseg = reorient(imageloc, segloc=truthloc)

            npimg = resize_to_nn(npimg, transpose=True).astype(np.int16)
            npseg = resize_to_nn(npseg, transpose=True).astype(np.uint8)

            print(imgs.shape)
            print(npimg.shape)

            imgs = np.vstack((imgs, npimg))
            segs = np.vstack((segs, npseg))

            print('imgs\t', imgs.shape)
            print('segs\t', segs.shape)

    return imgs, segs

def plot_histogram(data, b=100, r=(-990,990)):
    counts, bin_edges = np.histogram(data, bins=b, range=r)
    plt.bar(bin_edges[:-1], counts, width=[0.8*(bin_edges[i+1]-bin_edges[i]) for i in range(len(bin_edges)-1)])
    plt.show()

imgs, segs = get_imgs()
plot_histogram(imgs,r=(-990,990))

liver_idx = segs > 0
tumor_idx = segs > 1
only_liver_idx = liver_idx * (1.0 - tumor_idx)

imgs_masked = imgs * liver_idx - 300*(1.0 - liver_idx)
plot_histogram(imgs_masked, r=(-200,400))

tumors_masked = imgs * tumor_idx - 300*(1.0 - tumor_idx)
plot_histogram(tumors_masked, r=(-200,400))

liver_masked = imgs * only_liver_idx - 300*(1.0-only_liver_idx)
plot_histogram(liver_masked, r=(-200,400))

