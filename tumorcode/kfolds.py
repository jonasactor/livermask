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
from setupmodel import GetSetupKfolds, GetDataDictionary
from trainmodel import TrainModel
from predictmodel import PredictModelFromNumpy
from mymetrics import dsc_int_3D

################################
# Perform K-fold validation
################################
def OneKfold(i=0, datadict=None):

    k = settings.options.kfolds

    modelloc = TrainModel(idfold=i) 
    (train_set,test_set) = GetSetupKfolds(settings.options.dbfile, k, i)
    for idtest in test_set:
        baseloc = '%s/%03d/%03d' % (settings.options.outdir, k, i)
        imgloc  = '%s/%s' % (settings.options.rootlocation, datadict[idtest]['image'])
        segloc  = '%s/%s' % (settings.options.rootlocation, datadict[idtest]['label'])
        outloc  = '%s/label-%04d.nii.gz' % (baseloc, idtest)
        sumscore = 0
        if settings.options.numepochs > 0 and settings.options.makepredictions: # doing K-fold prediction as I train

            imagepredict = nib.load(imgloc)
            imageheader  = imagepredict.header
            numpypredict = imagepredict.get_data().astype(settings.IMG_DTYPE)
            allseg       = nib.load(segloc).get_data().astype(settings.SEG_DTYPE)

            liver_idx = allseg > 0
            tumor_idx = allseg > 1

            seg_liver = np.zeros_like(allseg)
            seg_liver[liver_idx] = 1

            seg_tumor = np.zeros_like(allseg)
            seg_tumor[tumor_idx] = 1

            image_liver = seg_liver*numpypredict - 100.0*(1.0 - seg_liver)
            image_liver = image_liver.astype(settings.IMG_DTYPE)

            predseg = PredictModelFromNumpy(model=modelloc, image=image_liver, imageheader=imageheader, outdir=outloc )
            score = dsc_int_3D(seg_tumor, predseg)
            sumscore += score
            print(idtest, "\t", score)
 
    print(k, " avg dice:\t", sumscore/len(test_set))

def Kfold():
    databaseinfo = GetDataDictionary(settings.options.dbfile)
    for iii in range(settings.options.kfolds):
        OneKfold(i=iii, datadict=databaseinfo)


