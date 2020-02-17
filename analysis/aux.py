import numpy as np
import os

import nibabel as nib

import keras
import keras.backend as K
import keras.losses
from keras import models
from keras.models import load_model

import tensorflow as tf


import sys
sys.setrecursionlimit(5000)
sys.path.append('liverhcc')
from mymetrics import dsc_l2, dsc_l2_3D, dsc_int, dsc_int_3D, dsc
from ista import ISTA
import preprocess


# out: np arrays of shape (nslices, ny, nx, 1)
def get_img(imgloc, segloc=None):
    npimg, _, npseg = preprocess.reorient(imgloc, segloc)
    npimg           = preprocess.resize_to_nn(npimg).astype(np.float32)
    npimg           = preprocess.window(npimg, -100,400)
    npimg           = preprocess.rescale(npimg, -100,400)
    npseg           = preprocess.resize_to_nn(npseg).astype(np.uint8)

    assert npimg.shape == npseg.shape
    return npimg[...,np.newaxis], npseg[...,np.newaxis]


def makeprediction(modelloc, imgloc, bs=16):
    im, sg = get_img(imgloc)
    m = load_model(modelloc, custom_dict={'dsc_l2':dsc_l2, 'dsc_int':dsc_int, 'ISTA':ISTA, 'dsc':dsc})
    pred = m.predict(im, batch_size=bs)
    pred_seg = (pred > 0.5).astype('uint8')
    return im, sg, pred, pred_seg


def makestatsmaps(modelloc, imgloc, outloc, ntrials=20):
    m = load_model(modelloc, custom_dict={'dsc_l2':dsc_l2, 'dsc_int':dsc_int, 'ISTA':ISTA, 'dsc':dsc})
    im, _ = get_img(imgloc)

    f = K.function([m.layers[0].input, K.learning_phase()],
                   [m.layers[-1].output])

    results = np.zeros(im.shape[0:-1] + (ntrials,))
    for jj in range(ntrials):
        results[...,jj] = f([im, 1])[0][...,0]

    pred_avg = results.mean(axis=-1)
    pred_var = results.var(axis=-1)
    pred_ent = np.zeros(pred_avg.shape)
    ent_idx0 = pred_avg > 0
    ent_idx1 = pred_avg < 1
    ent_idx  = np.logical_and(ent_idx0, ent_idx1)
    pred_ent[ent_idx] = - 1.0*np.multiply(       pred_avg[ent_idx], np.log(       pred_avg[ent_idx])) \
                        - 1.0*np.multiply( 1.0 - pred_avg[ent_idx], np.log( 1.0 - pred_avg[ent_idx]))

    save_as_nifti(pred_avg, outloc+'/pred-avg.nii.gz')
    save_as_nifti(pred_var, outloc+'/pred-var.nii.gz')
    save_as_nifti(pred_ent, outloc+'/pred-ent.nii.gz')



# in: img, a np array of shape (nslices, ny, nx, 1) or (nslices, ny, nx)
def save_as_nifti(img, outloc, resize=False):
    if len(img.shape)==4:
        imout = img[...,0]
    else:
        imout = img
    if resize:
        imout = preprocess.resize_to_original(imout)
    imnii = nib.Nifti1Image(imout, None )
    imnii.to_filename(outloc)

# in: img, a np array of shape (nslices, ny, nx, 1)
def compute_dsc_scores(y_true, y_pred):
    dims = y_true.shape
    dsc_l2_2D  = [None]*dims[0]
    dsc_int_2D = [None]*dims[0]

    dsc3D_l2  = dsc_l2_3D(y_true, y_pred)
    dsc3D_int = dsc_int_3D(y_true, y_pred)

    for iii in range(dims[0]):
        this_y_true = y_true[iii,...]
        this_y_pred = y_pred[iii,...]
        dsc_l2_2D[iii]  = dsc_l2(this_y_true[np.newaxis,...], this_y_pred[np.newaxis,...])
        dsc_int_2D[iii] = dsc_int(this_y_true[np.newaxis,...], this_y_pred[np.newaxis,...])

    return dsc3D_l2, dsc3D_int, dsc_l2_2D, dsc_int_2D



