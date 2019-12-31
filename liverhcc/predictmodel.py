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
from mymetrics import dsc, dsc_l2, l1
import preprocess

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
    origheader   = imagepredict.header
    imageaffine0 = imagepredict.affine
    imagepredict = nib.as_closest_canonical(imagepredict)
    imageaffine1 = imagepredict.affine
    print('\t\treoriented from ', nib.orientations.aff2axcodes(imageaffine0), ' to ', nib.orientations.aff2axcodes(imageaffine1))
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

    resizepredict = preprocess.window(resizepredict, settings.options.hu_lb, settings.options.hu_ub)
    resizepredict = preprocess.rescale(resizepredict, settings.options.hu_lb, settings.options.hu_ub)

    segout_float = loaded_model.predict( resizepredict[...,np.newaxis] )
    segout_int   = (segout_float[...,0] >= settings.options.segthreshold).astype(settings.SEG_DTYPE)

    segin_windowed = skimage.transform.resize(resizepredict,
            (nslice,settings._globalexpectedpixel,settings._globalexpectedpixel),
            order=0,
            preserve_range=True,
            mode='constant').transpose(2,1,0)
    segin_windowed_img = nib.Nifti1Image(segin_windowed, None, header=origheader)
    segin_windowed_img.to_filename( outdir.replace('.nii.gz', '-in-windowed.nii.gz') )

    segout_float_resize = skimage.transform.resize(segout_float[...,0],
            (nslice,settings._globalexpectedpixel,settings._globalexpectedpixel),
            order=0,
            preserve_range=True,
            mode='constant').transpose(2,1,0)
    segout_float_img = nib.Nifti1Image(segout_float_resize, None, header=origheader)
    segout_float_img.to_filename( outdir.replace('.nii.gz', '-predtumorfloat.nii.gz') )

    segout_int_resize = skimage.transform.resize(segout_int[...,0],
            (nslice,settings._globalexpectedpixel,settings._globalexpectedpixel),
            order=0,
            preserve_range=True,
            mode='constant').transpose(2,1,0)
    segout_int_img = nib.Nifti1Image(segout_int_resize, None, header=origheader)
    segout_int_img.to_filename( outdir.replace('.nii.gz', '-predtumorseg.nii.gz') )

    return segout_int_resize



############################
# apply model to numpy data
############################
###
### image argument needs
### to be np array 
### input is alreday rescaled
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


    resizepredict = preprocess.window(resizepredict, settings.options.hu_lb, settings.options.hu_ub)
    resizepredict = preprocess.rescale(resizepredict, settings.options.hu_lb, settings.options.hu_ub)

    segin_windowed = skimage.transform.resize(resizepredict,
            (nslice,settings._globalexpectedpixel,settings._globalexpectedpixel),
            order=0,
            preserve_range=True,
            mode='constant').transpose(2,1,0)
    segin_windowed_img = nib.Nifti1Image(segin_windowed, None, header=imageheader)
    segin_windowed_img.to_filename( outdir.replace('.nii.gz', '-in-windowed.nii.gz') )

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

    return segout_int_resize, segout_float_resize


############################
# test dropout + variation
############################
###
### image argument needs
### to be np array 
###
##########################
def PredictDropoutFromNumpy(model=None, image=None, imageheader=None, outdir=None):

    if model is None:
        model = settings.options.predictmodel
    if outdir is None:
        outdir = settings.options.segmentation

  
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    
    numpypredict = image
 
    # save unprocesed image_in
    img_in_nii = nib.Nifti1Image(image, None, header=imageheader)
    img_in_nii.to_filename( outdir.replace('.nii.gz', '-imgin.nii.gz') )

    assert numpypredict.shape[0:2] == (settings._globalexpectedpixel,settings._globalexpectedpixel)
    nslice = numpypredict.shape[2]

    # preprocessing
    resizepredict = skimage.transform.resize(numpypredict,
            (settings.options.trainingresample,settings.options.trainingresample,nslice ),
            order=0,
            preserve_range=True,
            mode='constant').astype(settings.IMG_DTYPE).transpose(2,1,0)

    resizepredict = preprocess.window( resizepredict, settings.options.hu_lb, settings.options.hu_ub)
    resizepredict = preprocess.rescale(resizepredict, settings.options.hu_lb, settings.options.hu_ub)

    # save preprocessed image_in
    segin_windowed = skimage.transform.resize(resizepredict,
            (nslice,settings._globalexpectedpixel,settings._globalexpectedpixel),
            order=0,
            preserve_range=True,
            mode='constant').transpose(2,1,0)
    segin_windowed_img = nib.Nifti1Image(segin_windowed, None, header=imageheader)
    segin_windowed_img.to_filename( outdir.replace('.nii.gz', '-imgin-windowed.nii.gz') )



    ###
    ### set up model
    ###

    opt          = GetOptimizer()
    lss, met     = GetLoss()
    loaded_model = get_unet()
    loaded_model.compile(loss=lss, metrics=met, optimizer=opt)
    loaded_model.load_weights(model)



    ###
    ### making baseline prediction and saving to file
    ###

    print('\tmaking baseline predictions...')

    segout_float = loaded_model.predict( resizepredict[...,np.newaxis] )
    segout_int   = (segout_float[...,0] >= settings.options.segthreshold).astype(settings.SEG_DTYPE)

    # save pred_float
    segout_float_resize = skimage.transform.resize(segout_float[...,0],
            (nslice,settings._globalexpectedpixel,settings._globalexpectedpixel),
            order=0,
            preserve_range=True,
            mode='constant').transpose(2,1,0)
    segout_float_img = nib.Nifti1Image(segout_float_resize, None, header=imageheader)
    segout_float_img.to_filename( outdir.replace('.nii.gz', '-predtumorfloat.nii.gz') )

    # save pred_int
    segout_int_resize = skimage.transform.resize(segout_int[...,0],
            (nslice,settings._globalexpectedpixel,settings._globalexpectedpixel),
            order=0,
            preserve_range=True,
            mode='constant').transpose(2,1,0)
    segout_int_img = nib.Nifti1Image(segout_int_resize, None, header=imageheader)
    segout_int_img.to_filename( outdir.replace('.nii.gz', '-predtumorseg.nii.gz') )



    ###
    ### making predictions using different Bernoulli draws for dropout
    ###

    print('\tmaking predictions with different dropouts trials...')

    f = K.function([loaded_model.layers[0].input, K.learning_phase()],
                   [loaded_model.layers[-1].output])

    results    = np.zeros(resizepredict.shape + (settings.options.ntrials,))
    for jj in range(settings.options.ntrials):
        results[...,jj] = f([resizepredict[...,np.newaxis], 1])[0][...,0]

    print('\tcalculating statistics...')

    pred_avg = results.mean(axis=-1)
    pred_var = results.var(axis=-1)
#    pred_ent = np.zeros(pred_avg.shape)
#    ent_idx  = 0 < pred_avg < 1
#    pred_ent[ent_idx] = -1*np.multiply(      pred_avg[ent_idx], np.log(      pred_avg[ent_idx])) \
#                        -1*np.multiply(1.0 - pred_avg[ent_idx], np.log(1.0 - pred_avg[ent_idx]))

    print('\tsaving trial statistics...')

    # save pred_avg
    pred_avg_resize = skimage.transform.resize(pred_avg,
            (nslice,settings._globalexpectedpixel,settings._globalexpectedpixel),
            order=0,
            preserve_range=True,
            mode='constant').transpose(2,1,0)
    pred_avg_img = nib.Nifti1Image(pred_avg_resize, None, header=imageheader)
    pred_avg_img.to_filename( outdir.replace('.nii.gz', '-pred-avg.nii.gz') )

    # save pred_var
    pred_var_resize = skimage.transform.resize(pred_var,
            (nslice,settings._globalexpectedpixel,settings._globalexpectedpixel),
            order=0,
            preserve_range=True,
            mode='constant').transpose(2,1,0)
    pred_var_img = nib.Nifti1Image(pred_var_resize, None, header=imageheader)
    pred_var_img.to_filename( outdir.replace('.nii.gz', '-pred-var.nii.gz') )
    
    # save pred_ent
#    pred_ent_resize = skimage.transform.resize(pred_ent,
#            (nslice,settings._globalexpectedpixel,settings._globalexpectedpixel),
#            order=0,
#            preserve_range=True,
#            mode='constant').transpose(2,1,0)
#    pred_ent_img = nib.Nifti1Image(pred_ent_resize, None, header=imageheader)
#    pred_ent_img.to_filename( outdir.replace('.nii.gz', '-pred-ent.nii.gz') )



    return segout_int_resize, segout_float_resize


############################
# test dropout + variation
############################
###
### image argument needs
### to be np array 
###
##########################
def PredictDropout():

    model  = settings.options.predictmodel
    image  = settings.options.predictimage
    outdir = settings.options.segmentation

    if not (model != None and image != None and outdir != None ):
        return

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    
    imagepredict = nib.load(image)
    origheader   = imagepredict.header
    imageaffine0 = imagepredict.affine
    imagepredict = nib.as_closest_canonical(imagepredict)
    imageaffine1 = imagepredict.affine
    print('\t\treoriented from ', nib.orientations.aff2axcodes(imageaffine0), ' to ', nib.orientations.aff2axcodes(imageaffine1))
    imageheader  = imagepredict.header
    numpypredict = imagepredict.get_data().astype(settings.IMG_DTYPE )
    assert numpypredict.shape[0:2] == (settings._globalexpectedpixel,settings._globalexpectedpixel)
    nslice = numpypredict.shape[2]

    # save unprocesed image_in
    img_in_nii = nib.Nifti1Image(numpypredict, None, header=origheader)
    img_in_nii.to_filename( outdir.replace('.nii', '-imgin.nii.gz') )

    # preprocessing
    resizepredict = skimage.transform.resize(numpypredict,
            (settings.options.trainingresample,settings.options.trainingresample,nslice ),
            order=0,
            preserve_range=True,
            mode='constant').astype(settings.IMG_DTYPE).transpose(2,1,0)

    resizepredict = preprocess.window( resizepredict, settings.options.hu_lb, settings.options.hu_ub)
    resizepredict = preprocess.rescale(resizepredict, settings.options.hu_lb, settings.options.hu_ub)

    # save preprocessed image_in
    segin_windowed = skimage.transform.resize(resizepredict,
            (nslice,settings._globalexpectedpixel,settings._globalexpectedpixel),
            order=0,
            preserve_range=True,
            mode='constant').transpose(2,1,0)
    segin_windowed_img = nib.Nifti1Image(segin_windowed, None, header=origheader)
    segin_windowed_img.to_filename( outdir.replace('.nii', '-imgin-windowed.nii.gz') )



    ###
    ### set up model
    ###

    loaded_model = load_model(model, custom_objects={'dsc_l2':dsc_l2, 'l1':l1, 'dsc':dsc, 'dsc_int':dsc})

    ###
    ### making baseline prediction and saving to file
    ###

    print('\tmaking baseline predictions...')

    segout_float = loaded_model.predict( resizepredict[...,np.newaxis] )
    segout_int   = (segout_float[...,0] >= settings.options.segthreshold).astype(settings.SEG_DTYPE)

    # save pred_float
    segout_float_resize = skimage.transform.resize(segout_float[...,0],
            (nslice,settings._globalexpectedpixel,settings._globalexpectedpixel),
            order=0,
            preserve_range=True,
            mode='constant').transpose(2,1,0)
    segout_float_img = nib.Nifti1Image(segout_float_resize, None, header=origheader)
    segout_float_img.to_filename( outdir.replace('.nii', '-pred-float.nii.gz') )

    # save pred_int
    segout_int_resize = skimage.transform.resize(segout_int[...,0],
            (nslice,settings._globalexpectedpixel,settings._globalexpectedpixel),
            order=0,
            preserve_range=True,
            mode='constant').transpose(2,1,0)
    segout_int_img = nib.Nifti1Image(segout_int_resize, None, header=origheader)
    segout_int_img.to_filename( outdir.replace('.nii', '-pred-seg.nii.gz') )



    ###
    ### making predictions using different Bernoulli draws for dropout
    ###

    print('\tmaking predictions with different dropouts trials...')

    f = K.function([loaded_model.layers[0].input, K.learning_phase()],
                   [loaded_model.layers[-1].output])

    results    = np.zeros(resizepredict.shape + (settings.options.ntrials,))
    for jj in range(settings.options.ntrials):
        results[...,jj] = f([resizepredict[...,np.newaxis], 1])[0][...,0]

    print('\tcalculating statistics...')

    pred_avg = results.mean(axis=-1)
    pred_var = results.var(axis=-1)
    pred_ent = np.zeros(pred_avg.shape)
#    ent_idx  = 0. < pred_avg < 1.
    ent_idx0 = pred_avg > 0
    ent_idx1 = pred_avg < 1
    ent_idx  = np.logical_and(ent_idx0, ent_idx1)
    pred_ent[ent_idx] = -1*np.multiply(      pred_avg[ent_idx], np.log(      pred_avg[ent_idx])) \
                        -1*np.multiply(1.0 - pred_avg[ent_idx], np.log(1.0 - pred_avg[ent_idx]))

    print('\tsaving trial statistics...')

    # save pred_avg
    pred_avg_resize = skimage.transform.resize(pred_avg,
            (nslice,settings._globalexpectedpixel,settings._globalexpectedpixel),
            order=0,
            preserve_range=True,
            mode='constant').transpose(2,1,0)
    pred_avg_img = nib.Nifti1Image(pred_avg_resize, None, header=origheader)
    pred_avg_img.to_filename( outdir.replace('.nii', '-pred-avg.nii.gz') )

    # save pred_var
    pred_var_resize = skimage.transform.resize(pred_var,
            (nslice,settings._globalexpectedpixel,settings._globalexpectedpixel),
            order=0,
            preserve_range=True,
            mode='constant').transpose(2,1,0)
    pred_var_img = nib.Nifti1Image(pred_var_resize, None, header=origheader)
    pred_var_img.to_filename( outdir.replace('.nii', '-pred-var.nii.gz') )
    
    # save pred_ent
    pred_ent_resize = skimage.transform.resize(pred_ent,
            (nslice,settings._globalexpectedpixel,settings._globalexpectedpixel),
            order=0,
            preserve_range=True,
            mode='constant').transpose(2,1,0)
    pred_ent_img = nib.Nifti1Image(pred_ent_resize, None, header=origheader)
    pred_ent_img.to_filename( outdir.replace('.nii', '-pred-ent.nii.gz') )



    return segout_int_resize, segout_float_resize




