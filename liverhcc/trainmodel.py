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
import preprocess

###
### Training: build NN model from anonymized data
###
def TrainModel(idfold=0):

  from setupmodel import GetSetupKfolds, GetCallbacks, GetOptimizer, GetLoss
  from buildmodel import get_unet

  ###
  ### set up output, logging, and callbacks
  ###
  kfolds = settings.options.kfolds

  logfileoutputdir= '%s/%03d/%03d' % (settings.options.outdir, kfolds, idfold)
  os.system ('mkdir -p ' + logfileoutputdir)
  os.system ('mkdir -p ' + logfileoutputdir + '/nii')
  os.system ('mkdir -p ' + logfileoutputdir + '/liver')
  print("Output to\t", logfileoutputdir)

  ###
  ### load data
  ###


  print('loading memory map db for large dataset')
  numpydatabase = np.load(settings._globalnpfile)
  (train_index,test_index,valid_index) = GetSetupKfolds(settings.options.dbfile, kfolds, idfold)

  print('copy data subsets into memory...')
  axialbounds = numpydatabase['axialliverbounds']
  dataidarray = numpydatabase['dataid']
  dbtrainindex = np.isin(dataidarray, train_index )
  dbtestindex  = np.isin(dataidarray, test_index  )
  dbvalidindex = np.isin(dataidarray, valid_index ) 
  subsetidx_train  = np.all( np.vstack((axialbounds , dbtrainindex)) , axis=0 )
  subsetidx_test   = np.all( np.vstack((axialbounds , dbtestindex )) , axis=0 )
  subsetidx_valid  = np.all( np.vstack((axialbounds , dbvalidindex)) , axis=0 )
  if np.sum(subsetidx_train) + np.sum(subsetidx_test) + np.sum(subsetidx_valid) != min(np.sum(axialbounds ),np.sum(dbtrainindex )) :
      raise("data error: slice numbers dont match")

  print('copy memory map from disk to RAM...')
  trainingsubset = numpydatabase[subsetidx_train]
  validsubset    = numpydatabase[subsetidx_valid]
  testsubset     = numpydatabase[subsetidx_test ]

#  trimg = trainingsubset['imagedata']
#  trseg = trainingsubset['truthdata']
#  vaimg = validsubset['imagedata']
#  vaseg = validsubset['truthdata']
#  teimg = testsubset['imagedata']
#  teseg = testsubset['truthdata']

#  trimg_img = nib.Nifti1Image(trimg, None)
#  trimg_img.to_filename( logfileoutputdir+'/nii/train-img.nii.gz')
#  vaimg_img = nib.Nifti1Image(vaimg, None)
#  vaimg_img.to_filename( logfileoutputdir+'/nii/valid-img.nii.gz')
#  teimg_img = nib.Nifti1Image(teimg, None)
#  teimg_img.to_filename( logfileoutputdir+'/nii/test-img.nii.gz')
#
#  trseg_img = nib.Nifti1Image(trseg, None)
#  trseg_img.to_filename( logfileoutputdir+'/nii/train-seg.nii.gz')
#  vaseg_img = nib.Nifti1Image(vaseg, None)
#  vaseg_img.to_filename( logfileoutputdir+'/nii/valid-seg.nii.gz')
#  teseg_img = nib.Nifti1Image(teseg, None)
#  teseg_img.to_filename( logfileoutputdir+'/nii/test-seg.nii.gz')

  np.random.seed(seed=0)
  np.random.shuffle(trainingsubset)

  ntrainslices = len(trainingsubset)
  nvalidslices = len(validsubset)

  x_train=trainingsubset['imagedata']
  y_train=trainingsubset['truthdata']

  x_valid=validsubset['imagedata']
  y_valid=validsubset['truthdata']

  print("\nkfolds : ", kfolds)
  print("idfold : ",   idfold)
  print("slices training   : ", ntrainslices)
  print("slices validation : ", nvalidslices)


  ###
  ### data preprocessing : applying liver mask
  ###
  y_train_typed = y_train.astype(settings.SEG_DTYPE)
  y_train_liver = preprocess.livermask(y_train_typed)

  x_train_typed = x_train
  x_train_typed = preprocess.window(x_train_typed, settings.options.hu_lb, settings.options.hu_ub)
  x_train_typed = preprocess.rescale(x_train_typed, settings.options.hu_lb, settings.options.hu_ub)

  y_valid_typed = y_valid.astype(settings.SEG_DTYPE)
  y_valid_liver = preprocess.livermask(y_valid_typed)

  x_valid_typed = x_valid
  x_valid_typed = preprocess.window(x_valid_typed, settings.options.hu_lb, settings.options.hu_ub)
  x_valid_typed = preprocess.rescale(x_valid_typed, settings.options.hu_lb, settings.options.hu_ub)



  ###
  ### create and run model
  ###
  opt                 = GetOptimizer()
  callbacks, modelloc = GetCallbacks(logfileoutputdir, "liver")
  lss, met            = GetLoss()
  model               = get_unet()
  model.compile(loss  = lss,
        metrics       = met,
        optimizer     = opt)

  print("\n\n\tlivermask training...\tModel parameters: {0:,}".format(model.count_params()))


  if settings.options.augment:
      train_datagen = ImageDataGenerator(
          brightness_range=[0.9,1.1],
          preprocessing_function=preprocess.post_augment,
      )
      train_maskgen = ImageDataGenerator(
              )
  else:
      train_datagen=ImageDataGenerator()
      train_maskgen=ImageDataGenerator()

  sd = 2 # arbitrary but fixed seed for ImageDataGenerators()
  dataflow = train_datagen.flow(x_train_typed[...,np.newaxis],
          batch_size=settings.options.trainingbatch,
          seed=sd,
          shuffle=True)
  maskflow = train_maskgen.flow(y_train_liver[...,np.newaxis],
          batch_size=settings.options.trainingbatch,
          seed=sd,
          shuffle=True)
  train_generator = zip(dataflow, maskflow)

#  train_generator = train_datagen.flow(x_train_typed[...,np.newaxis],
#          y=y_train_liver[...,np.newaxis],
#          batch_size=settings.options.trainingbatch,
#          seed=sd,
#          shuffle=True)

  valid_datagen = ImageDataGenerator()
  valid_maskgen = ImageDataGenerator()

  validdataflow = valid_datagen.flow(x_valid_typed[...,np.newaxis],
          batch_size=settings.options.validationbatch,
          seed=sd,
          shuffle=True)
  validmaskflow = valid_maskgen.flow(y_valid_liver[...,np.newaxis],
          batch_size=settings.options.validationbatch,
          seed=sd,
          shuffle=True)
  valid_generator = zip(validdataflow,validmaskflow)

###
### visualize augmentation
###
#
#  import matplotlib
#  matplotlib.use('TkAgg')
#  from matplotlib import pyplot as plt
#  for i in range(8):
#      plt.subplot(4,4,2*i + 1)
#      imbatch = dataflow.next()
#      sgbatch = maskflow.next()
#      imaug = imbatch[0][:,:,0]
#      sgaug = sgbatch[0][:,:,0]
#      plt.imshow(imaug)
#      plt.subplot(4,4,2*i + 2)
#      plt.imshow(sgaug)
#  plt.show()
#  return
#

  history_liver = model.fit_generator(
                        train_generator,
                        steps_per_epoch = ntrainslices / settings.options.trainingbatch,
                        epochs=settings.options.numepochs,
                        validation_data=valid_generator,
                        callbacks=callbacks,
                        shuffle=True,
                        validation_steps= nvalidslices / settings.options.validationbatch,
                        )



  ###
  ### make predicions on validation set
  ###
  print("\n\n\tapplying models...")
  y_pred_float = model.predict( x_valid_typed[...,np.newaxis] )
  y_pred_seg   = (y_pred_float[...,0] >= settings.options.segthreshold).astype(settings.SEG_DTYPE)

  print("\tsaving to file...")
  trueinnii     = nib.Nifti1Image(x_valid,       None)
  truesegnii    = nib.Nifti1Image(y_valid,       None)
#  windownii     = nib.Nifti1Image(x_valid_typed, None)
  truelivernii  = nib.Nifti1Image(y_valid_liver, None)
  predsegnii    = nib.Nifti1Image(y_pred_seg, None )
  predfloatnii  = nib.Nifti1Image(y_pred_float, None)
 
  trueinnii.to_filename(    logfileoutputdir+'/nii/trueimg.nii.gz')
  truesegnii.to_filename(   logfileoutputdir+'/nii/trueseg.nii.gz')
#  windownii.to_filename(    logfileoutputdir+'/nii/windowedimg.nii.gz')
  truelivernii.to_filename( logfileoutputdir+'/nii/trueliver.nii.gz')
  predsegnii.to_filename(   logfileoutputdir+'/nii/predtumorseg.nii.gz')
  predfloatnii.to_filename( logfileoutputdir+'/nii/predtumorfloat.nii.gz')

  print("\done saving.")
  return modelloc


