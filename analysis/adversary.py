import numpy as np
import os
from optparse import OptionParser
import matplotlib as mptlib
mptlib.use('TkAgg')
import matplotlib.pyplot as plt

import nibabel as nib

import keras
import keras.backend as K
import keras.losses
from keras import models
from keras import layers
from keras.models import load_model
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

import tensorflow as tf

import sys
sys.setrecursionlimit(5000)
sys.path.append('/rsrch1/ip/jacctor/livermask/liverhcc')
from mymetrics import dsc_l2, dsc, dsc_int, l1
from ista import ISTA
import preprocess



###
### set options
###
parser = OptionParser()
parser.add_option( "--model",
        action="store", dest="model", default=None,
        help="model location", metavar="PATH")
parser.add_option( "--outdir",
        action="store", dest="outdir", default="./",
        help="out location", metavar="PATH")
parser.add_option( "--imgloc",
        action="store", dest="imgloc", default='/rsrch1/ip/jacctor/LiTS/LiTS/TrainingBatch2/volume-124.nii',
        help="img location", metavar="PATH.nii")
parser.add_option( "--segloc",
        action="store", dest="segloc", default='/rsrch1/ip/jacctor/LiTS/LiTS/TrainingBatch2/segmentation-124.nii',
        help="seg location", metavar="PATH.nii")
parser.add_option( "--delta",
        type="float", dest="delta", default=0.1,
        help="perturbation", metavar="float")
(options, args) = parser.parse_args()



###
### start a session - will need same session to link K to tf
###

session = tf.Session()
K.set_session(session)

###
### get data
###

def get_img(imgloc, segloc=None):
    npimg, _, npseg = preprocess.reorient(imgloc, segloc)
    npimg           = preprocess.resize_to_nn(npimg).astype(np.float32)
    npimg           = preprocess.window(npimg, -100,400)
    npimg           = preprocess.rescale(npimg, -100,400)
    npseg           = preprocess.resize_to_nn(npseg).astype(np.uint8)

    print(npimg.shape)
    print(npseg.shape)
    assert npimg.shape == npseg.shape
    return npimg[...,np.newaxis], npseg[...,np.newaxis]
###
### WARNING: can't return only one slice for evaluation if there are 2 gpus, since it will try 
###          to send one slice to each gpu and then not send anything to gpu:1
###          otherwise error: check failed work_element_count > 0 (0 vs 0)
#    midslice = npimg[int(npimg.shape[0] / 2),:,:]
#    midseg   = npseg[int(npimg.shape[0] / 2),:,:]
#    return midslice[np.newaxis,:,:,np.newaxis], midseg[np.newaxis,:,:,np.newaxis]

x_test, y_test = get_img(options.imgloc, options.segloc)

IMG_DTYPE = np.int16
SEG_DTYPE = np.uint8

_globalexpectedpixel=512
_nx = 256
_ny = 256


###
### load model and perform adversarial attack
###

net = load_model(options.model, custom_objects={'dsc_l2':dsc_l2, 'dsc':dsc, 'dsc_int':dsc_int, 'ISTA':ISTA})

def l2loss(y_true, y_pred):
    return keras.losses.mean_squared_error(y_true,y_pred)

def fgsm(ximg, yseg, net, loss=l2loss, eps=options.delta):

    print(ximg.shape)
    print(yseg.shape)

    x_adv   = np.zeros_like(ximg)
    perturb = np.zeros_like(ximg)

    nslices = ximg.shape[0]
    sss = [32*s for s in range(nslices//32)]
    if nslices%32 > 1:
        sss.append(nslices)
    print(sss)
    for s in range(len(sss)-1):

        srt = sss[s]
        end = sss[s+1]
        x = ximg[srt:end,...]
        y = yseg[srt:end,...]
        #x = x[np.newaxis,...]
        #y = y[np.newaxis,...]

        yy = y.astype(np.float32)
        loss_vec = loss(yy, net.output)
        grad = K.gradients(loss_vec, net.input)
        gradfunc = K.function([net.input], grad)
        grad_at_x = gradfunc([x])
        perturb_this = eps*np.sign(grad_at_x[0])
        x_adv_this = x + perturb_this

        x_adv[srt:end,...] = x_adv_this
        perturb[srt:end,...] = perturb_this

    return x_adv, perturb


losses = ['l2', 'dsc']
lossfxns = [l2loss, dsc_l2]

for iii in range(len(losses)):

    lll = losses[iii]
    lfn = lossfxns[iii]

    x_adv, perturb = fgsm(x_test, y_test, net, eps=options.delta, loss=lfn)

    x_adv_orig = x_adv[...,0]
    x_adv_orig_nii = nib.Nifti1Image(x_adv_orig , None )
    x_adv_orig_nii.to_filename(options.outdir+lll+'/'+'img-adv.nii.gz')

    perturb_orig = perturb[...,0]
    perturb_orig_nii = nib.Nifti1Image(perturb_orig, None )
    perturb_orig_nii.to_filename(options.outdir+lll+'/'+'perturb.nii.gz')

    x_test_orig = x_test[...,0]
    x_test_orig_nii = nib.Nifti1Image(x_test_orig , None )
    x_test_orig_nii.to_filename(options.outdir+lll+'/'+'img.nii.gz')

    y_test_orig = y_test[...,0]
    y_test_orig_nii = nib.Nifti1Image(y_test_orig , None )
    y_test_orig_nii.to_filename(options.outdir+lll+'/'+'seg.nii.gz')

    y_pred = net.predict(x_test, batch_size=16)
    y_pred_orig = y_pred[...,0]
    y_pred_orig_nii = nib.Nifti1Image(y_pred_orig, None )
    y_pred_orig_nii.to_filename(options.outdir+lll+'/'+'seg-pred.nii.gz')

    y_adv = net.predict(x_adv, batch_size=16)
    y_adv_orig = y_adv[...,0]
    y_adv_orig_nii = nib.Nifti1Image(y_adv_orig , None )
    y_adv_orig_nii.to_filename(options.outdir+lll+'/'+'seg-pred-adv.nii.gz')


