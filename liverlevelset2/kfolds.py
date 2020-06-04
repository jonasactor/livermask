import numpy as np
import csv
import sys
import os
import json
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
from setupmodel import GetSetupKfolds, GetDataDictionary
from trainmodel import TrainModel
from predictmodel import PredictModel, PredictDropout
from mymetrics import dsc_int_3D, dsc_l2_3D

################################
# Perform K-fold validation
################################
def OneKfold(i=0, datadict=None):

    k = settings.options.kfolds

    modelloc = TrainModel(idfold=i) 
    (train_set,test_set,valid_set) = GetSetupKfolds(settings.options.dbfile, k, i)

    sumscore = 0
    sumscorefloat = 0
    for idtest in test_set:
        baseloc = '%s/%03d/%03d' % (settings.options.outdir, k, i)
        imgloc  = '%s/%s' % (settings.options.rootlocation, datadict[idtest]['image'])
        segloc  = '%s/%s' % (settings.options.rootlocation, datadict[idtest]['label'])
        outloc  = '%s/label-%04d.nii.gz' % (baseloc, idtest)
        if settings.options.numepochs > 0 and (settings.options.makepredictions or settings.options.makedropoutmap):


            if settings.options.makepredictions:
                predseg, predfloat = PredictModel(  model=modelloc, image=imgloc, outdir=outloc) 
            else:
                predseg, predfloat = PredictDropout(model=modelloc, image=imgloc, outdir=outloc, seg=segloc)

#            seg = nib.load(segloc).get_data().astype(settings.SEG_DTYPE)
#            seg_liver = preprocess.livermask(seg)
#
#            score_float = dsc_l2_3D(seg_liver, predfloat)
#            sumscorefloat += score_float
#            print(idtest, "\t", score_float)
#
#    print(k, " avg dice:\t", sumscorefloat/len(test_set)) 

def Kfold():
    databaseinfo = GetDataDictionary(settings.options.dbfile)
    for iii in range(settings.options.kfolds):
        OneKfold(i=iii, datadict=databaseinfo)


