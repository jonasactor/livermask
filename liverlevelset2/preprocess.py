import numpy as np
import nibabel as nib
import settings
from scipy import ndimage
import skimage.transform
from skimage.measure import label

# cut off img intensities
# img : npy array
# lb  : desired lower bound
# ub  : desired upper bound
def window(img, lb, ub):
    too_low  = img <= lb
    too_high = img >= ub
    img[too_low]  = lb
    img[too_high] = ub
    return img

# rescale img to [-1,   1] if no augmentation
# rescale img to [ 0, 255] if augmentation
# img : npy array
# lb  : known lower bound in image
# ub  : known upper bound in image
def rescale(img, lb, ub):
    rs = img.astype(settings.FLOAT_DTYPE)
    rs = 2.*(rs - lb)/(ub-lb) - 1.0 
    rs = rs.astype(settings.FLOAT_DTYPE)
    return rs

# turn liver+tumor seg into just liver seg
def livermask(seg):
    liver_idx = seg > 0
    liver = np.zeros_like(seg)
    liver[liver_idx] = 1
    return liver.astype(settings.IMG_DTYPE)

# turn liver+tumor seg into just tumor seg
def tumormask(seg):
    tumor_idx = seg > 1
    tumor = np.zeros_like(seg)
    tumor[tumor_idx] = 1
    return tumor.astype(settings.IMG_DTYPE)


# reorient NIFTI files into RAS+
# takes care to perform same reorientation for both image and segmentation
# takes image header as truth if segmentation and image headers differ
def reorient(imgloc, segloc=None):

    imagedata   = nib.load(imgloc)
    orig_affine = imagedata.affine
    orig_header = imagedata.header
    imagedata   = nib.as_closest_canonical(imagedata)
    img_affine  = imagedata.affine
    numpyimage = imagedata.get_data().astype(settings.IMG_DTYPE)
    numpyseg   = None
    print('image :    ', nib.orientations.aff2axcodes(orig_affine), ' to ', nib.orientations.aff2axcodes(img_affine))

    if segloc is not None:
        segdata    = nib.load(segloc)
        old_affine = segdata.affine
        segdata    = nib.as_closest_canonical(segdata)
        seg_affine = segdata.affine
        if not np.allclose(seg_affine, img_affine):
            segcopy = nib.load(segloc).get_data()
            copy_header = orig_header.copy()
            segdata = nib.nifti1.Nifti1Image(segcopy, orig_affine, header=copy_header)
            segdata = nib.as_closest_canonical(segdata)
            seg_affine = segdata.affine
        print('seg   :    ', nib.orientations.aff2axcodes(old_affine), ' to ', nib.orientations.aff2axcodes(seg_affine))
        numpyseg = segdata.get_data().astype(settings.SEG_DTYPE)

    return numpyimage, orig_header, numpyseg

# sample down to nn's expected input size
def resize_to_nn(img,transpose=True):
    expected = skimage.transform.resize(img,
            (settings.options.trainingresample,settings.options.trainingresample,img.shape[2]),
            order=0,
            mode='constant',
            preserve_range=True)
    if transpose:
        expected = expected.transpose(2,1,0)
    return expected

# return to original size
def resize_to_original(img,transpose=True):
    real = skimage.transform.resize(img,
            (img.shape[0],settings._globalexpectedpixel,settings._globalexpectedpixel),
            order=0,
            mode='constant',
            preserve_range=True)
    if transpose:
        real = real.transpose(2,1,0)
    return real

# returns largest connected component of {0,1} binary segmentation image
# in : img \in {0,1}^n_pixels
def largest_connected_component(img):
    labels = label(img)
    assert ( labels.max() != 0)
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

def post_augment(img):
    return rescale(img, 0, 255)
