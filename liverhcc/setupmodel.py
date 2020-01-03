import numpy as np
import csv
import sys
import os
import json
import keras
from keras.models import model_from_json, load_model
from keras.utils.np_utils import to_categorical
import keras.backend as K
from keras.callbacks import TensorBoard, TerminateOnNaN, ModelCheckpoint
from keras.callbacks import Callback as CallbackBase
from keras.preprocessing.image import ImageDataGenerator
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

# build data base from CSV file
def GetDataDictionary(floc):
  CSVDictionary = {}
  with open(floc, 'r') as csvfile:
    myreader = csv.DictReader(csvfile, delimiter=',')
    for row in myreader:
       CSVDictionary[int( row['dataid'])]  =  {'image':row['image'], 'label':row['label']}
  return CSVDictionary


# setup kfolds
def GetSetupKfolds(floc, numfolds, idfold):
  # get id from setupfiles
  dataidsfull = []
  with open(floc, 'r') as csvfile:
    myreader = csv.DictReader(csvfile, delimiter=',')
    for row in myreader:
       dataidsfull.append( int( row['dataid']))
  if (numfolds < idfold or numfolds < 1):
     raise("data input error")
  # split in folds
  if (numfolds > 1):
     kf = KFold(n_splits=numfolds)
     allkfolds   = [ (train_index, test_index) for train_index, test_index in kf.split(dataidsfull )]
     train_index = allkfolds[idfold][0]
     test_index  = allkfolds[idfold][1]
  else:
     train_index = np.array(dataidsfull )
     test_index  = None
  print("kfold: \t",numfolds)
  print("idfold: \t", idfold)
  print("train_index:\t", train_index)
  print("test_index:\t",  test_index)
  return (train_index,test_index)




###
### Build the numpy database of images:
### preprocess database and store to disk
###

def BuildDB():
  # create  custom data frame database type
  mydatabasetype = [('dataid', int),
     ('axialliverbounds',bool),
     ('axialtumorbounds',bool),
     ('imagepath','S128'),
     ('imagedata','(%d,%d)int16' % (settings.options.trainingresample, settings.options.trainingresample)),
     ('truthpath','S128'),
     ('truthdata','(%d,%d)uint8' % (settings.options.trainingresample, settings.options.trainingresample))] 

  # initialize empty dataframe
  numpydatabase = np.empty(0, dtype=mydatabasetype  )

  # load all data from csv
  totalnslice = 0
  with open(settings.options.dbfile, 'r') as csvfile:
    myreader = csv.DictReader(csvfile, delimiter=',')
    for row in myreader:
      imagelocation = '%s/%s' % (settings.options.rootlocation,row['image'])
      truthlocation = '%s/%s' % (settings.options.rootlocation,row['label'])
      print(imagelocation,truthlocation )

      numpyimage, orig_header, numpytruth  = preprocess.reorient(imagelocation, segloc=truthlocation)


      # error check
      assert numpyimage.shape[0:2] == (settings._globalexpectedpixel,settings._globalexpectedpixel)
      nslice = numpyimage.shape[2]
      assert numpytruth.shape[0:2] == (settings._globalexpectedpixel,settings._globalexpectedpixel)
      assert nslice  == numpytruth.shape[2]

      resimage = preprocess.resize_to_nn(numpyimage, transpose=False).astype(settings.IMG_DTYPE)
      restruth = preprocess.resize_to_nn(numpytruth, transpose=False).astype(settings.SEG_DTYPE)


      # bounding box for each label
      if( np.max(restruth) ==1 ) :
        (liverboundingbox,)  = ndimage.find_objects(restruth)
        tumorboundingbox  = None
      else:
        (liverboundingbox,tumorboundingbox) = ndimage.find_objects(restruth)

      if( nslice  == restruth.shape[2]):
        # custom data type to subset
        datamatrix = np.zeros(nslice  , dtype=mydatabasetype )

        # custom data type to subset
        datamatrix ['dataid']                         = np.repeat(row['dataid'], nslice )
        # id the slices within the bounding box
        axialliverbounds                              = np.repeat(False, nslice )
        axialtumorbounds                              = np.repeat(False, nslice )
        axialliverbounds[liverboundingbox[2]]         = True
        if (tumorboundingbox != None):
          axialtumorbounds[tumorboundingbox[2]]       = True
        datamatrix ['axialliverbounds']               = axialliverbounds
        datamatrix ['axialtumorbounds']               = axialtumorbounds
        datamatrix ['imagepath']                      = np.repeat(imagelocation, nslice )
        datamatrix ['truthpath']                      = np.repeat(truthlocation, nslice )
        datamatrix ['imagedata']                      = resimage.transpose(2,1,0)
        datamatrix ['truthdata']                      = restruth.transpose(2,1,0)
        numpydatabase = np.hstack((numpydatabase,datamatrix))
        # count total slice for QA
        totalnslice = totalnslice + nslice
      else:
        print('training data error image[2] = %d , truth[2] = %d ' % (nslice,restruth.shape[2]))

  # save numpy array to disk
  np.save( settings._globalnpfile, numpydatabase)




###
### training option helper functions
###

def GetCallbacks(logfileoutputdir, stage):
  logdir   = logfileoutputdir+"/"+stage
  filename = logfileoutputdir+"/"+stage+"/modelunet.h5"
  logname  = logfileoutputdir+"/"+stage+"/log.csv"
  if settings.options.with_hvd:
      callbacks = [ hvd.callbacks.BroadcastGlobalVariablesCallback(0),
                    hvd.callbacks.MetricAverageCallback(),
                    hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=1),
                    keras.callbacks.TerminateOnNaN()         ]
      if hvd.rank() == 0:
          callbacks += [ keras.callbacks.ModelCheckpoint(filepath=filename, verbose=1, save_best_only=True),
                         keras.callbacks.CSVLogger(logname),
                         keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=0, write_graph=True, write_images=False)      ]
  else:
      callbacks = [ keras.callbacks.TerminateOnNaN(),
                    keras.callbacks.CSVLogger(logname),
                    keras.callbacks.ModelCheckpoint(filepath=filename, verbose=1, save_best_only=True),  
                    keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=0, write_graph=True, write_images=False)  ] 
  return callbacks, filename

def GetOptimizer():
  if settings.options.with_hvd:
      if settings.options.trainingsolver=="adam":
          opt = keras.optimizers.Adam(lr=settings.options.lr*hvd.size())
      elif settings.options.trainingsolver=="adadelta":
          opt = keras.optimizers.Adadelta(1.0*hvd.size())
      elif settings.options.trainingsolver=="nadam":
          opt = keras.optimizers.Nadam(0.002*hvd.size())
      elif settings.options.trainingsolver=="sgd":
          opt = keras.optimizers.SGD(0.01*hvd.size())
      else:
          raise Exception("horovod-enabled optimizer not selected")
      opt = hvd.DistributedOptimizer(opt)
  else:
      if settings.options.trainingsolver=="adam":
          opt = keras.optimizers.Adam(lr=settings.options.lr)
      elif settings.options.trainingsolver=="adadelta":
          opt = keras.optimizers.Adadelta(1.0)
      elif settings.options.trainingsolver=="nadam":
          opt = keras.optimizers.Nadam(0.002)
      elif settings.options.trainingsolver=="sgd":
          opt = keras.optimizers.SGD(0.01)
      else:
          opt = settings.options.trainingsolver
  return opt

def GetLoss():

  from mymetrics import dsc, dsc_l2, dsc_int, dsc_int_3D, l1

  lss = dsc_l2
  met = [dsc_l2, l1, dsc, dsc_int]

  return lss, met




