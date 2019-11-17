import numpy as np
import csv
import sys
import os
import json
import keras
from keras.layers import Input, Conv2D, UpSampling2D, Lambda, SpatialDropout2D, Dense, Layer, Activation, BatchNormalization, MaxPool2D, concatenate, LocallyConnected2D
from keras.models import Model, Sequential
from keras.models import model_from_json, load_model
from keras.utils import multi_gpu_model
from keras.utils.np_utils import to_categorical
import keras.backend as K
from keras.callbacks import TensorBoard, TerminateOnNaN, ModelCheckpoint
from keras.callbacks import Callback as CallbackBase
from keras.preprocessing.image import ImageDataGenerator
from keras.initializers import Constant
import nibabel as nib
from scipy import ndimage
from sklearn.model_selection import KFold
import skimage.transform
import tensorflow as tf
import matplotlib as mptlib
mptlib.use('TkAgg')
import matplotlib.pyplot as plt


import settings
from setupmodel import GetOptimizer, GetLoss
from buildmodel import get_unet


#############################
# apply model to NIFTI image
#############################
def PredictModel():
  
  model  = settings.options.predictmodel
  image  = settings.options.predictimage
  outdir = settings.options.segmentation


  if (model != None and image != None and outdir != None ):
  
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    
    imagepredict = nib.load(image)
    imageheader  = imagepredict.header
    numpypredict = imagepredict.get_data().astype(settings.IMG_DTYPE )
    assert numpypredict.shape[0:2] == (settings._globalexpectedpixel,settings._globalexpectedpixel)
    nslice = numpypredict.shape[2]

    resizepredict = skimage.transform.resize(numpypredict,
            (settings.options.trainingresample,settings.options.trainingresample,nslice ),
            order=0,
            preserve_range=True,
            mode='constant').astype(settings.IMG_DTYPE).transpose(2,1,0)


    opt          = GetOptimizer()
    lss, met     = GetLoss()
    loaded_model = get_unet()
    loaded_model.compile(loss=lss, metrics=met, optimizer=opt)
    loaded_model.load_weights(model)

    segout_float = loaded_model.predict( resizepredict[...,np.newaxis] )
    segout_int   = (segout_float[...,0] >= settings.options.segthreshold).astype(settings.SEG_DTYPE)

    segout_float_resize = skimage.transform.resize(segout_float[...,0],
            (nslice,settings._globalexpectedpixel,settings._globalexpectedpixel),
            order=0,
            preserve_range=True,
            mode='constant').transpose(2,1,0)
    segout_float_img = nib.Nifti1Image(segout_float_resize, None, header=imageheader)
    segout_float_img.to_filename( outdir.replace('.nii.gz', '-predtumorfloat.nii.gz') )

    segout_int_resize = skimage.transform.resize(segout_int[...,0],
            (nslice,settings._globalexpectedpixel,settings._globalexpectedpixel),
            order=0,
            preserve_range=True,
            mode='constant').transpose(2,1,0)
    segout_int_img = nib.Nifti1Image(segout_int_resize, None, header=imageheader)
    segout_int_img.to_filename( outdir.replace('.nii.gz', '-predtumorseg.nii.gz') )

    return segout_int_resize



############################
# apply model to numpy data
############################
###
### image argument needs
### to be np array masked
### for liver values only
###
##########################
def PredictModelFromNumpy(model=None, image=None, imageheader=None, outdir=None):

    if model is None:
        model = settings.options.predictmodel
    if outdir is None:
        outdir = settings.options.segmentation

  
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    
    numpypredict = image
    assert numpypredict.shape[0:2] == (settings._globalexpectedpixel,settings._globalexpectedpixel)
    nslice = numpypredict.shape[2]

    resizepredict = skimage.transform.resize(numpypredict,
            (settings.options.trainingresample,settings.options.trainingresample,nslice ),
            order=0,
            preserve_range=True,
            mode='constant').astype(settings.IMG_DTYPE).transpose(2,1,0)

    opt          = GetOptimizer()
    lss, met     = GetLoss()
    loaded_model = get_unet()
    loaded_model.compile(loss=lss, metrics=met, optimizer=opt)
    loaded_model.load_weights(model)

    
    img_in_nii = nib.Nifti1Image(image, None, header=imageheader)
    img_in_nii.to_filename( outdir.replace('.nii.gz', '-imgin.nii.gz') )


    segout_float = loaded_model.predict( resizepredict[...,np.newaxis] )
    segout_int   = (segout_float[...,0] >= settings.options.segthreshold).astype(settings.SEG_DTYPE)

    segout_float_resize = skimage.transform.resize(segout_float[...,0],
            (nslice,settings._globalexpectedpixel,settings._globalexpectedpixel),
            order=0,
            preserve_range=True,
            mode='constant').transpose(2,1,0)
    segout_float_img = nib.Nifti1Image(segout_float_resize, None, header=imageheader)
    segout_float_img.to_filename( outdir.replace('.nii.gz', '-predtumorfloat.nii.gz') )

    segout_int_resize = skimage.transform.resize(segout_int[...,0],
            (nslice,settings._globalexpectedpixel,settings._globalexpectedpixel),
            order=0,
            preserve_range=True,
            mode='constant').transpose(2,1,0)
    segout_int_img = nib.Nifti1Image(segout_int_resize, None, header=imageheader)
    segout_int_img.to_filename( outdir.replace('.nii.gz', '-predtumorseg.nii.gz') )

    return segout_int_resize


