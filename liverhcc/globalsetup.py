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


def perform_setup(options):
    
    sys.setrecursionlimit(5000)

    if options.with_hvd:
        import horovod.keras as hvd
        hvd.init()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        if options.gpu > 1:
            devlist = '0'
            for i in range(1,options.gpu):
                devlist += ','+str(i)
            config.gpu_options.visible_device_list = devlist
        else:
            config.gpu_options.visible_device_list = str(hvd.local_rank())
        K.set_session(tf.Session(config=config))


    global _globalnpfile
    global _globalexpectedpixel
    global INT_DTYPE
    global SEG_DTYPE
    global _nx
    global _ny


    # raw dicom data is usually short int (2bytes) datatype
    # labels are usually uchar (1byte)
    IMG_DTYPE = np.int16
    SEG_DTYPE = np.uint8

    _globalnpfile = options.dbfile.replace('.csv','%d.npy' % options.trainingresample )
    _globalexpectedpixel=512
    _nx = options.trainingresample
    _ny = options.trainingresample

    return IMG_DTYPE, SEG_DTYPE, _globalnpfile, _globalexpectedpixel, _nx, _ny
