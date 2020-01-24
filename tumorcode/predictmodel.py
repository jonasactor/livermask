import numpy as np
import csv
import sys
import os
import json
import keras
from keras.models import Model, Sequential
from keras.models import model_from_json, load_model
from keras.utils.np_utils import to_categorical
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
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
def PredictModel(model=settings.options.predictmodel, image=settings.options.predictimage, imageheader=None, outdir=settings.options.segmentation):
  
  if (model != None and image != None and outdir != None ):
  
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    
    numpypredict, origheader, _ = preprocess.reorient(image)

    assert numpypredict.shape[0:2] == (settings._globalexpectedpixel,settings._globalexpectedpixel)

    resizepredict = preprocess.resize_to_nn(numpypredict)
    resizepredict = preprocess.window(resizepredict, settings.options.hu_lb, settings.options.hu_ub)
    resizepredict = preprocess.rescale(resizepredict, settings.options.hu_lb, settings.options.hu_ub)

    ###
    ### set up model
    ###

    loaded_model = load_model(model, custom_objects={'dsc_l2':dsc_l2, 'l1':l1, 'dsc':dsc, 'dsc_int':dsc})

    segout_float = loaded_model.predict( resizepredict[...,np.newaxis] )[...,0]
    segout_int   = (segout_float >= settings.options.segthreshold).astype(settings.SEG_DTYPE)

    segin_windowed = preprocess.resize_to_original(resizepredict)
    segin_windowed_img = nib.Nifti1Image(segin_windowed, None, header=origheader)
    segin_windowed_img.to_filename( outdir.replace('.nii', '-imgin-windowed.nii') )

    segout_float_resize = preprocess.resize_to_original(segout_float)
    segout_float_img = nib.Nifti1Image(segout_float_resize, None, header=origheader)
    segout_float_img.to_filename( outdir.replace('.nii', '-pred-float.nii') )

    segout_int_resize = preprocess.resize_to_original(segout_int)
    segout_int_img = nib.Nifti1Image(segout_int_resize, None, header=origheader)
    segout_int_img.to_filename( outdir.replace('.nii', '-pred-seg.nii') )

    return segout_float_resize, segout_int_resize


############################
# test dropout + variation
############################
def PredictDropout(model=settings.options.predictmodel, image=settings.options.predictimage, outdir=settings.options.segmentation):

    if not (model != None and image != None and outdir != None ):
        return

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    
    numpypredict, origheader, _ = preprocess.reorient(image)

    assert numpypredict.shape[0:2] == (settings._globalexpectedpixel,settings._globalexpectedpixel)

    resizepredict = preprocess.resize_to_nn(numpypredict)
    resizepredict = preprocess.window(resizepredict, settings.options.hu_lb, settings.options.hu_ub)
    resizepredict = preprocess.rescale(resizepredict, settings.options.hu_lb, settings.options.hu_ub)

    opt          = GetOptimizer()
    lss, met     = GetLoss()
    loaded_model = get_unet()
    loaded_model.compile(loss=lss, metrics=met, optimizer=opt)
    loaded_model.load_weights(model)

    # save unprocesed image_in
    img_in_nii = nib.Nifti1Image(numpypredict, None, header=origheader)
    img_in_nii.to_filename( outdir.replace('.nii', '-imgin.nii') )

    # save preprocessed image_in
    segin_windowed = preprocess.resize_to_original(resizepredict)
    segin_windowed_img = nib.Nifti1Image(segin_windowed, None, header=origheader)
    segin_windowed_img.to_filename( outdir.replace('.nii', '-imgin-windowed.nii') )



    ###
    ### set up model
    ###

    loaded_model = load_model(model, custom_objects={'dsc_l2':dsc_l2, 'l1':l1, 'dsc':dsc, 'dsc_int':dsc})

    ###
    ### making baseline prediction and saving to file
    ###

    print('\tmaking baseline predictions...')

    segout_float = loaded_model.predict( resizepredict[...,np.newaxis] )[...,0]
    segout_int   = (segout_float >= settings.options.segthreshold).astype(settings.SEG_DTYPE)

    segout_float_resize = preprocess.resize_to_original(segout_float)
    segout_float_img = nib.Nifti1Image(segout_float_resize, None, header=origheader)
    segout_float_img.to_filename( outdir.replace('.nii', '-pred-float.nii') )

    segout_int_resize = preprocess.resize_to_original(segout_int)
    segout_int_img = nib.Nifti1Image(segout_int_resize, None, header=origheader)
    segout_int_img.to_filename( outdir.replace('.nii', '-pred-seg.nii') )



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
    ent_idx0 = pred_avg > 0
    ent_idx1 = pred_avg < 1
    ent_idx  = np.logical_and(ent_idx0, ent_idx1)
    pred_ent[ent_idx] = -1*np.multiply(      pred_avg[ent_idx], np.log(      pred_avg[ent_idx])) \
                        -1*np.multiply(1.0 - pred_avg[ent_idx], np.log(1.0 - pred_avg[ent_idx]))

    print('\tsaving trial statistics...')

    # save pred_avg
    pred_avg_resize = preprocess.resize_to_original(pred_avg)
    pred_avg_img = nib.Nifti1Image(pred_avg_resize, None, header=origheader)
    pred_avg_img.to_filename( outdir.replace('.nii', '-pred-avg.nii') )

    # save pred_var
    pred_var_resize = preprocess.resize_to_original(pred_var)
    pred_var_img = nib.Nifti1Image(pred_var_resize, None, header=origheader)
    pred_var_img.to_filename( outdir.replace('.nii', '-pred-var.nii') )
    
    # save pred_ent
    pred_ent_resize = preprocess.resize_to_original(pred_ent)
    pred_ent_img = nib.Nifti1Image(pred_ent_resize, None, header=origheader)
    pred_ent_img.to_filename( outdir.replace('.nii', '-pred-ent.nii') )

    return segout_int_resize, segout_float_resize

