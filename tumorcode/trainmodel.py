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
from optparse import OptionParser # TODO update to ArgParser (python2 --> python3)
import nibabel as nib
from scipy import ndimage
from sklearn.model_selection import KFold
import skimage.transform
import tensorflow as tf
import matplotlib as mptlib
mptlib.use('TkAgg')
import matplotlib.pyplot as plt

import settings 


###
### Training: build NN model from anonymized data
###
def TrainModel(idfold=0):

  from setupmodel import GetSetupKfolds, GetCallbacks, GetOptimizer, GetLoss
  from buildmodel import get_unet

  ###
  ### load data
  ###

  kfolds = settings.options.kfolds

  print('loading memory map db for large dataset')
  numpydatabase = np.load(settings._globalnpfile)
  (train_index,test_index) = GetSetupKfolds(settings.options.dbfile, kfolds, idfold)

  print('copy data subsets into memory...')
  axialbounds = numpydatabase['axialtumorbounds']
  dataidarray = numpydatabase['dataid']
  dbtrainindex= np.isin(dataidarray, train_index )
  dbtestindex = np.isin(dataidarray, test_index  )
  subsetidx_train  = np.all( np.vstack((axialbounds , dbtrainindex)) , axis=0 )
  subsetidx_test   = np.all( np.vstack((axialbounds , dbtestindex )) , axis=0 )
  if np.sum(subsetidx_train) + np.sum(subsetidx_test) != min(np.sum(axialbounds ),np.sum(dbtrainindex )) :
      raise("data error: slice numbers dont match")

  print('copy memory map from disk to RAM...')
  trainingsubset = numpydatabase[subsetidx_train]

  np.random.seed(seed=0)
  np.random.shuffle(trainingsubset)
  totnslice = len(trainingsubset)

  x_train=trainingsubset['imagedata']
  y_train=trainingsubset['truthdata']

  slicesplit        = int(0.9 * totnslice)
  TRAINING_SLICES   = slice(         0, slicesplit)
  VALIDATION_SLICES = slice(slicesplit, totnslice )

  print("\nkfolds : ", kfolds)
  print("idfold : ",   idfold)
  print("slices in kfold   : ", totnslice)
  print("slices training   : ", slicesplit)
  print("slices validation : ", totnslice - slicesplit)
  try:
      print("slices testing    : ", len(numpydatabase[subsetidx_test]))
  except:
      print("slices testing    : 0")


  ###
  ### data preprocessing : applying liver mask
  ###
  y_train_typed = y_train.astype(settings.SEG_DTYPE)

  liver_idx = y_train_typed > 0
  y_train_liver = np.zeros_like(y_train_typed)
  y_train_liver[liver_idx] = 1

  tumor_idx = y_train_typed > 1
  y_train_tumor = np.zeros_like(y_train_typed)
  y_train_tumor[tumor_idx] = 1

  x_masked = x_train * y_train_liver - 100.0*(1.0 - y_train_liver)
  x_masked = x_masked.astype(settings.IMG_DTYPE)


  ###
  ### set up output, logging, and callbacks
  ###
  logfileoutputdir= '%s/%03d/%03d' % (settings.options.outdir, kfolds, idfold)
  os.system ('mkdir -p ' + logfileoutputdir)
  os.system ('mkdir -p ' + logfileoutputdir + '/nii')
  os.system ('mkdir -p ' + logfileoutputdir + '/tumor')
  print("Output to\t", logfileoutputdir)



  ###
  ### create and run model
  ###
  opt                 = GetOptimizer()
  callbacks, modelloc = GetCallbacks(logfileoutputdir, "tumor")
  lss, met            = GetLoss()
  model               = get_unet()
  model.compile(loss  = lss,
        metrics       = met,
        optimizer     = opt)

  print("\n\n\tlivermask training...\tModel parameters: {0:,}".format(model.count_params()))


  if settings.options.augment:
      train_datagen = ImageDataGenerator(
          brightness_range=[0.95,1.05],
          width_shift_range=[-0.1,0.1],
          height_shift_range=[-0.1,0.1],
          horizontal_flip=True,
          vertical_flip=True,
          zoom_range=0.1,
          fill_mode='nearest',
          )
  else:
      train_datagen=ImageDataGenerator()

  test_datagen = ImageDataGenerator()

  train_generator = train_datagen.flow(x_masked[TRAINING_SLICES,:,:,np.newaxis],
                        y_train_tumor[TRAINING_SLICES,:,:,np.newaxis],
                        batch_size=settings.options.trainingbatch)
  test_generator = test_datagen.flow(x_masked[VALIDATION_SLICES,:,:,np.newaxis],
                        y_train_tumor[VALIDATION_SLICES,:,:,np.newaxis],
                        batch_size=settings.options.validationbatch)
  history_liver = model.fit_generator(
                        train_generator,
                        steps_per_epoch= slicesplit / settings.options.trainingbatch,
                        epochs=settings.options.numepochs,
                        validation_data=test_generator,
                        callbacks=callbacks)



  ###
  ### make predicions on validation set
  ###
  print("\n\n\tapplying models...")
  y_pred_float = model.predict( x_masked[VALIDATION_SLICES,:,:,np.newaxis] )
  y_pred_seg   = (y_pred_float[...,0] >= settings.options.segthreshold).astype(settings.SEG_DTYPE)

  print("\tsaving to file...")
  trueinnii     = nib.Nifti1Image(x_train      [VALIDATION_SLICES,:,:] , None )
  truesegnii    = nib.Nifti1Image(y_train      [VALIDATION_SLICES,:,:] , None )
  truelivernii  = nib.Nifti1Image(y_train_liver[VALIDATION_SLICES,:,:] , None )
  truetumornii  = nib.Nifti1Image(y_train_tumor[VALIDATION_SLICES,:,:] , None )
  predsegnii    = nib.Nifti1Image(y_pred_seg, None )
  predfloatnii  = nib.Nifti1Image(y_pred_float, None)
 
  trueinnii.to_filename(    logfileoutputdir+'/nii/trueimg.nii.gz')
  truesegnii.to_filename(   logfileoutputdir+'/nii/truseg.nii.gz')
  truelivernii.to_filename( logfileoutputdir+'/nii/trueliver.nii.gz')
  truetumornii.to_filename( logfileoutputdir+'/nii/truetumor.nii.gz')
  predsegnii.to_filename(   logfileoutputdir+'/nii/predtumorseg.nii.gz')
  predfloatnii.to_filename( logfileoutputdir+'/nii/predtumorfloat.nii.gz')

  print("\done saving.")
  return modelloc


