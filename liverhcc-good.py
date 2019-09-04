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
from optparse import OptionParser # TODO update to ArgParser (python2 --> python3)
import nibabel as nib
from scipy import ndimage
from sklearn.model_selection import KFold
import skimage.transform
import tensorflow as tf
import matplotlib as mptlib
mptlib.use('TkAgg')
import matplotlib.pyplot as plt

sys.setrecursionlimit(5000)

# setup command line parser to control execution
parser = OptionParser()
parser.add_option( "--hvd",
                  action="store_true", dest="with_hvd", default=False,
                  help="use horovod for multicore parallelism")
parser.add_option( "--gpu",
                  type="int", dest="gpu", default=0,
                  help="number of gpus", metavar="int")
parser.add_option( "--builddb",
                  action="store_true", dest="builddb", default=False,
                  help="load all training data into npy", metavar="FILE")
parser.add_option( "--trainmodel",
                  action="store_true", dest="trainmodel", default=False,
                  help="train model on all data", metavar="bool")
parser.add_option( "--predictmodel",
                  action="store", dest="predictmodel", default=None,
                  help="model weights (.h5) for prediction", metavar="Path")
parser.add_option( "--predictimage",
                  action="store", dest="predictimage", default=None,
                  help="image to segment", metavar="Path")
parser.add_option( "--segmentation",
                  action="store", dest="segmentation", default=None,
                  help="location for seg prediction output ", metavar="Path")
parser.add_option( "--trainingsolver",
                  action="store", dest="trainingsolver", default='adam',
                  help="setup info", metavar="string")
parser.add_option( "--dbfile",
                  action="store", dest="dbfile", default="./trainingdata.csv",
                  help="training data file", metavar="string")
parser.add_option( "--trainingresample",
                  type="int", dest="trainingresample", default=256,
                  help="resample so that model prediction occurs at this resolution", metavar="int")
parser.add_option( "--trainingbatchliver",
                  type="int", dest="trainingbatchliver", default=20,
                  help="batch size", metavar="int")
parser.add_option( "--validationbatch",
                  type="int", dest="validationbatch", default=20,
                  help="batch size", metavar="int")
parser.add_option( "--kfolds",
                  type="int", dest="kfolds", default=1,
                  help="perform kfold prediction with k folds", metavar="int")
parser.add_option( "--idfold",
                  type="int", dest="idfold", default=-1,
                  help="individual fold for k folds", metavar="int")
parser.add_option( "--rootlocation",
                  action="store", dest="rootlocation", default='/rsrch1/ip/jacctor/LiTS/LiTS',
                  help="root location for images for training", metavar="Path")
parser.add_option("--numepochs",
                  type="int", dest="numepochs_liver", default=10,
                  help="number of epochs for training", metavar="int")
parser.add_option("--outdir",
                  action="store", dest="outdir", default='./',
                  help="directory for output", metavar="Path")
parser.add_option( "--skip",
                  action="store_true", dest="skip", default=True,
                  help="skip connections in UNet", metavar="bool")
parser.add_option( "--fanout",
                  action="store_true", dest="fanout", default=True,
                  help="fan out as UNet gets deeper (more filters at deeper levels)", metavar="bool")
parser.add_option( "--batchnorm",
                  action="store_true", dest="batchnorm", default=False,
                  help="use batch normalization in UNet", metavar="bool")
parser.add_option( "--depth",
                  type="int", dest="depth", default=3,
                  help="number of down steps to UNet", metavar="int")
parser.add_option( "--filters",
                  type="int", dest="filters", default=16,
                  help="number of filters for output of CNN layer", metavar="int")
parser.add_option( "--activation",
                  action="store", dest="activation", default='relu',
                  help="activation function", metavar="string")
parser.add_option( "--segthreshold",
                  type="float", dest="segthreshold", default=0.5,
                  help="cutoff for binary segmentation from real-valued output", metavar="float")
(options, args) = parser.parse_args()

# raw dicom data is usually short int (2bytes) datatype
# labels are usually uchar (1byte)
IMG_DTYPE = np.int16
SEG_DTYPE = np.uint8

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


_globalnpfile = options.dbfile.replace('.csv','%d.npy' % options.trainingresample )
_globalexpectedpixel=512
_nx = options.trainingresample
_ny = options.trainingresample
print('database file: %s ' % _globalnpfile )


# build data base from CSV file
def GetDataDictionary():
  CSVDictionary = {}
  with open(options.dbfile, 'r') as csvfile:
    myreader = csv.DictReader(csvfile, delimiter=',')
    for row in myreader:
       CSVDictionary[int( row['dataid'])]  =  {'image':row['image'], 'label':row['label']}
  return CSVDictionary


# setup kfolds
def GetSetupKfolds(numfolds,idfold):
  # get id from setupfiles
  dataidsfull = []
  with open(options.dbfile, 'r') as csvfile:
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


def addConvBNSequential(model, filters=32, kernel_size=(3,3), batch_norm=True, activation='prelu', dropout=0.):
    if batch_norm:
          model = BatchNormalization()(model)
    if dropout > 0.0:
          model = SpatialDropout2D(dropout)(model)
    model = Conv2D(filters=filters, kernel_size=kernel_size, padding='same', activation=activation)(model)
    return model

def module_down(model, filters=16, kernel_size=(3,3), activation='prelu', batch_norm=True, dropout=0.5):
    model = addConvBNSequential(model, filters=filters, kernel_size=kernel_size, activation=activation, batch_norm=batch_norm)
    model = addConvBNSequential(model, filters=filters, kernel_size=kernel_size, activation=activation, batch_norm=batch_norm)
    model = MaxPool2D()(model)
    return model

def module_up(model, filters=16, kernel_size=(3,3), activation='prelu', batch_norm=True, dropout=0.5):
    model = addConvBNSequential(model, filters=filters, kernel_size=kernel_size, activation=activation, batch_norm=batch_norm, dropout=dropout)
    model = addConvBNSequential(model, filters=filters, kernel_size=kernel_size, activation=activation, batch_norm=batch_norm, dropout=dropout)
    model = UpSampling2D()(model)
    return model

def module_mid(model, depth, filters=16, kernel_size=(3,3), activation='prelu', batch_norm=True, dropout=0.5):
    if options.fanout and depth<2:
        filters = filters*2
    if depth==0:
        model = addConvBNSequential(model, filters=filters, kernel_size=kernel_size, activation=activation, batch_norm=batch_norm, dropout=dropout)
        model = addConvBNSequential(model, filters=filters, kernel_size=kernel_size, activation=activation, batch_norm=batch_norm, dropout=dropout)
        return model
    else:
        m_down = module_down( model,                filters=filters, kernel_size=kernel_size, activation=activation, batch_norm=batch_norm)
        m_mid  = module_mid (m_down, depth=depth-1, filters=filters, kernel_size=kernel_size, activation=activation, batch_norm=batch_norm, dropout=dropout)
        m_up   = module_up  ( m_mid,                filters=filters, kernel_size=kernel_size, activation=activation, batch_norm=batch_norm, dropout=dropout)
        if options.skip:
            m_up = concatenate([model, m_up])
        return m_up

def get_unet(_depth=3, _filters=16, _kernel_size=(3,3), _activation='prelu', _final_layer='sigmoid', _batch_norm=True, _dropout=0.5, _num_classes=1):
    if options.gpu > 1:
        with tf.device('/cpu:0'):
            layer_in  = Input(shape=(_ny,_nx,1))
            layer_adj = Activation('linear')(layer_in)
            layer_mid = module_mid(layer_adj, depth=_depth, filters=_filters, kernel_size=_kernel_size, activation=_activation, batch_norm=_batch_norm, dropout=_dropout)
            layer_out = Dense(_num_classes, activation=_final_layer, use_bias=True)(layer_mid)
            model = Model(inputs=layer_in, outputs=layer_out)
            return multi_gpu_model(model, gpus=options.gpu)
    else:
        layer_in  = Input(shape=(_ny,_nx,1))
        layer_adj = Activation('linear')(layer_in)
        layer_mid = module_mid(layer_adj, depth=_depth, filters=_filters, kernel_size=_kernel_size, activation=_activation, batch_norm=_batch_norm, dropout=_dropout)
        layer_out = Dense(_num_classes, activation=_final_layer, use_bias=True)(layer_mid)
        model = Model(inputs=layer_in, outputs=layer_out)
        return model

# dsc = 1 - dsc_as_l2
def dsc_l2(y_true, y_pred, smooth=0.00001):
    num = K.sum(K.square(y_true - y_pred), axis=(1,2)) + smooth
    den = K.sum(K.square(y_true), axis=(1,2)) + K.sum(K.square(y_pred), axis=(1,2)) + smooth
    return num/den


##########################
# preprocess database and store to disk
##########################
def BuildDB():
  # create  custom data frame database type
  mydatabasetype = [('dataid', int),
     ('axialliverbounds',bool),
     ('axialtumorbounds',bool),
     ('imagepath','S128'),
     ('imagedata','(%d,%d)int16' %(options.trainingresample,options.trainingresample)),
     ('truthpath','S128'),
     ('truthdata','(%d,%d)uint8' % (options.trainingresample,options.trainingresample))] 

  # initialize empty dataframe
  numpydatabase = np.empty(0, dtype=mydatabasetype  )

  # load all data from csv
  totalnslice = 0
  with open(options.dbfile, 'r') as csvfile:
    myreader = csv.DictReader(csvfile, delimiter=',')
    for row in myreader:
      imagelocation = '%s/%s' % (options.rootlocation,row['image'])
      truthlocation = '%s/%s' % (options.rootlocation,row['label'])
      print(imagelocation,truthlocation )

      # load nifti file
      imagedata = nib.load(imagelocation )
      numpyimage= imagedata.get_data().astype(IMG_DTYPE )
      # error check
      assert numpyimage.shape[0:2] == (_globalexpectedpixel,_globalexpectedpixel)
      nslice = numpyimage.shape[2]
      resimage=skimage.transform.resize(numpyimage,
            (options.trainingresample,options.trainingresample,nslice),
            order=0,
            mode='constant',
            preserve_range=True).astype(IMG_DTYPE)

      # load nifti file
      truthdata = nib.load(truthlocation )
      numpytruth= truthdata.get_data().astype(SEG_DTYPE)
      # error check
      assert numpytruth.shape[0:2] == (_globalexpectedpixel,_globalexpectedpixel)
      assert nslice  == numpytruth.shape[2]
      restruth=skimage.transform.resize(numpytruth,
              (options.trainingresample,options.trainingresample,nslice),
              order=0,
              mode='constant',
              preserve_range=True).astype(SEG_DTYPE)

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
  np.save( _globalnpfile, numpydatabase)

def GetCallbacks(logfileoutputdir, stage):
  logdir   = logfileoutputdir+"/"+stage
  filename = logfileoutputdir+"/"+stage+"/modelunet.h5"
  logname  = logfileoutputdir+"/"+stage+"/log.csv"
  if options.with_hvd:
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
  if options.with_hvd:
      if options.trainingsolver=="adam":
          opt = keras.optimizers.Adam(lr=0.001*hvd.size())
      elif options.trainingsolver=="adadelta":
          opt = keras.optimizers.Adadelta(1.0*hvd.size())
      elif options.trainingsolver=="nadam":
          opt = keras.optimizers.Nadam(0.002*hvd.size())
      elif options.trainingsolver=="sgd":
          opt = keras.optimizers.SGD(0.01*hvd.size())
      else:
          raise Exception("horovod-enabled optimizer not selected")
      opt = hvd.DistributedOptimizer(opt)
  else:
      if options.trainingsolver=="adam":
          opt = keras.optimizers.Adam(lr=0.01)
      elif options.trainingsolver=="adadelta":
          opt = keras.optimizers.Adadelta(1.0)
      elif options.trainingsolver=="nadam":
          opt = keras.optimizers.Nadam(0.2)
      elif options.trainingsolver=="sgd":
          opt = keras.optimizers.SGD(0.01)
      else:
          opt = options.trainingsolver
  return opt



##########################
# build NN model from anonymized data
##########################
def TrainModel(kfolds=options.kfolds,idfold=0):


  ###
  ### load data
  ###

  print('loading memory map db for large dataset')
  numpydatabase = np.load(_globalnpfile)
  (train_index,test_index) = GetSetupKfolds(kfolds,idfold)

  print('copy data subsets into memory...')
  axialbounds = numpydatabase['axialliverbounds']
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
  print("slices total      : ", totnslice)
  print("slices training   : ", slicesplit)
  print("slices validation : ", totnslice - slicesplit)
  try:
      print("slices testing    : ", len(numpydatabase[subsetidx_test]))
  except:
      print("slices testing    : 0")


  ###
  ### data preprocessing
  ###
  y_train_typed = y_train.astype(SEG_DTYPE)
  t_max = np.max(y_train_typed)

  liver_idx = y_train_typed > 0
  y_train_liver = np.zeros_like(y_train_typed)
  y_train_liver[liver_idx] = 1



  ###
  ### set up output, logging, and callbacks
  ###
  logfileoutputdir= '%s/%03d/%03d' % (options.outdir, kfolds, idfold)
  os.system ('mkdir -p ' + logfileoutputdir)
  os.system ('mkdir -p ' + logfileoutputdir + '/nii')
  os.system ('mkdir -p ' + logfileoutputdir + '/liver')
  print("Output to\t", logfileoutputdir)

  

  ###
  ### create and run model for Liver mask
  ###
  opt = GetOptimizer()
  callbacks, livermodelloc = GetCallbacks(logfileoutputdir, "liver")
  model_liver = get_unet(_depth=options.depth, _filters=options.filters, _activation=options.activation, _num_classes=1, _batch_norm=options.batchnorm)
  model_liver.compile(loss=dsc_l2,
        metrics=["binary_crossentropy"],
        optimizer=opt)

  print("\n\n\tlivermask training...\tModel parameters: {0:,}".format(model_liver.count_params()))
 
  history_liver = model_liver.fit( x_train[TRAINING_SLICES,:,:,np.newaxis], 
                             y_train_liver[TRAINING_SLICES,:,:,np.newaxis],
                             validation_data=(x_train[VALIDATION_SLICES,:,:,np.newaxis], y_train_liver[VALIDATION_SLICES,:,:,np.newaxis]),
                             callbacks = callbacks,
                             batch_size=options.trainingbatchliver,
                             epochs=options.numepochs_liver)


  ###
  ### make predicions on validation set
  ###
  print("\n\n\tapplying models...")
  y_pred_liver_float = model_liver.predict( x_train[VALIDATION_SLICES,:,:,np.newaxis] )
  y_pred_liver       = (y_pred_liver_float[...,0] >= options.segthreshold).astype(SEG_DTYPE)

  print("\tsaving to file...")
  trueinnii     = nib.Nifti1Image(x_train      [VALIDATION_SLICES,:,:] , None )
  truelivernii  = nib.Nifti1Image(y_train_liver[VALIDATION_SLICES,:,:] , None )
  predlivernii  = nib.Nifti1Image(y_pred_liver, None )
  predfloatnii  = nib.Nifti1Image(y_pred_liver_float, None)
 
  trueinnii.to_filename(    logfileoutputdir+'/nii/trueimg-raw.nii.gz')
  truelivernii.to_filename( logfileoutputdir+'/nii/trueseg-liver.nii.gz')
  predlivernii.to_filename( logfileoutputdir+'/nii/predseg-liver.nii.gz')
  predfloatnii.to_filename( logfileoutputdir+'/nii/predfloat-liver.nii.gz')

  return livermodelloc


##########################
# apply model to new data
##########################
def PredictModel(livermodel=options.predictmodel, image=options.predictimage, outdir=options.segmentation):
  if (model != None and image != None and outdir != None ):
  
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    
    imagepredict = nib.load(image)
    imageheader  = imagepredict.header
    numpypredict = imagepredict.get_data().astype(IMG_DTYPE )
    assert numpypredict.shape[0:2] == (_globalexpectedpixel,_globalexpectedpixel)
    nslice = numpypredict.shape[2]
    print(nslice)

    resizepredict = skimage.transform.resize(numpypredict,
            (options.trainingresample,options.trainingresample,nslice ),
            order=0,
            preserve_range=True,
            mode='constant').astype(IMG_DTYPE).transpose(2,1,0)

    opt = GetOptimizer()
    loaded_liver_model = get_unet(_depth=options.depth, _filters=options.filters, _activation=options.activation, _num_classes=1, _batch_norm=options.batchnorm)
    loaded_liver_model.compile(loss=dsc_l2,
          metrics=["binary_crossentropy"],
          optimizer=opt)
    loaded_liver_model.load_weights(livermodel)

    liversegout = loaded_liver_model.predict( resizepredict[...,np.newaxis] )
    liversegout = (liversegout[...,0] >= options.segthreshold).astype(SEG_DTYPE)

    liversegout_resize = skimage.transform.resize(liversegout[...,0],
            (nslice,_globalexpectedpixel,_globalexpectedpixel),
            order=0,
            preserve_range=True,
            mode='constant').transpose(2,1,0)
    liversegout_img = nib.Nifti1Image(liversegout_resize, None, header=imageheader)
    liversegout_img.to_filename( outdir.replace('.nii.gz', '-liverseg.nii.gz') )


################################
# Perform K-fold validation
################################
def OneKfold(k=options.kfolds, i=0, datadict=None):
    livermodelloc, tumormodelloc = TrainModel(kfolds=k, idfold=i) 
    (train_set,test_set) = GetSetupKfolds(k,i)
    for idtest in test_set:
        baseloc = '%s/%03d/%03d' % (options.outdir, k, i)
        imgloc  = '%s/%s' % (options.rootlocation, datadict[idtest]['image'])
        outloc  = '%s/label-%04d.nii.gz' % (baseloc, idtest) 
        if options.numepochs > 0:
            PredictModel(livermodel=livermodelloc, image=imgloc, outdir=outloc )

def Kfold(kkk):
    databaseinfo = GetDataDictionary()
    for iii in range(kkk):
        OneKfold(k=kkk, i=iii, datadict=databaseinfo)


if options.builddb:
    BuildDB()
if options.kfolds > 1:
    if options.idfold > -1:
        databaseinfo = GetDataDictionary()
        OneKfold(k=options.kfolds, i=options.idfold, datadict=databaseinfo)
    else:
        Kfold(options.kfolds)
if options.trainmodel: # no kfolds, i.e. k=1
    TrainModel(kfolds=1,idfold=0)
if options.predictmodel:
    PredictModel()
if ( (not options.builddb) and (not options.trainmodel) and (not options.predictmodel) and (options.kfolds == 1)):
    parser.print_help()
