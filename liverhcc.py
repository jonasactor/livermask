import numpy as np
import csv
import os
import json
import keras
from keras.layers import Input, Conv2D, UpSampling2D, Lambda, SpatialDropout2D, Dense, Layer, Activation, BatchNormalization, MaxPool2D, concatenate
from keras.models import Model, Sequential
from keras.models import model_from_json, load_model
from keras.utils.np_utils import to_categorical
import keras.backend as K
from keras.callbacks import TensorBoard, TerminateOnNaN, ModelCheckpoint
from keras.callbacks import Callback as CallbackBase
from keras.preprocessing.image import ImageDataGenerator
from optparse import OptionParser
import nibabel as nib
from scipy import ndimage
from sklearn.model_selection import KFold
import skimage.transform
import tensorflow as tf
#import horovod.keras as hvd
import sys

sys.setrecursionlimit(5000)

# setup command line parser to control execution
parser = OptionParser()
parser.add_option( "--hvd",
                  action="store_true", dest="with_hvd", default=False,
                  help="use horovod for parallelism")
parser.add_option( "--builddb",
                  action="store_true", dest="builddb", default=False,
                  help="load all training data into npy", metavar="FILE")
parser.add_option( "--trainmodel",
                  action="store_true", dest="trainmodel", default=False,
                  help="train model on all data", metavar="FILE")
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
                  action="store", dest="trainingsolver", default='nadam',
                  help="setup info", metavar="string")
parser.add_option( "--dbfile",
                  action="store", dest="dbfile", default="./trainingdata.csv",
                  help="training data file", metavar="string")
parser.add_option( "--trainingresample",
                  type="int", dest="trainingresample", default=256,
                  help="resample so that model prediction occurs at this resolution", metavar="int")
parser.add_option( "--trainingbatch",
                  type="int", dest="trainingbatch", default=4,
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
                  help="root location for images for training", metavar="string")
parser.add_option("--numepochs",
                  type="int", dest="numepochs", default=10,
                  help="number of epochs for training", metavar="int")
parser.add_option("--outdir",
                  action="store", dest="outdir", default='./',
                  help="directory for output", metavar="string")
parser.add_option( "--skip",
                  action="store_true", dest="skip", default=False,
                  help="skip connections in UNet", metavar="FILE")
parser.add_option( "--fanout",
                  action="store_true", dest="fanout", default=False,
                  help="fan out as UNet gets deeper (more filters at deeper levels)", metavar="FILE")
parser.add_option( "--batchnorm",
                  action="store_true", dest="batchnorm", default=False,
                  help="use batch normalization in UNet", metavar="FILE")
parser.add_option( "--depth",
                  type="int", dest="depth", default=2,
                  help="number of down steps to UNet", metavar="int")
parser.add_option( "--filters",
                  type="int", dest="filters", default=16,
                  help="number of filters for output of CNN layer", metavar="int")
parser.add_option( "--activation",
                  action="store", dest="activation", default='relu',
                  help="activation function", metavar="string")
(options, args) = parser.parse_args()

# raw dicom data is usually short int (2bytes) datatype
# labels are usually uchar (1byte)
IMG_DTYPE = np.int16
SEG_DTYPE = np.uint8

if options.with_hvd:
    hvd.init()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
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


def module_down(model, filters=16, kernel_size=(3,3), activation='prelu', batch_norm=True):
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
    if options.fanout:
        filters = filters*2
    if depth==0:
        model = addConvBNSequential(model, filters=filters, kernel_size=kernel_size, activation=activation, batch_norm=batch_norm, dropout=dropout)
        model = addConvBNSequential(model, filters=filters, kernel_size=kernel_size, activation=activation, batch_norm=batch_norm, dropout=dropout)
        return model
    else:
        m_down = module_down( model,                filters=filters, kernel_size=kernel_size, activation=activation, batch_norm=batch_norm)
        m_mid  = module_mid( m_down, depth=depth-1, filters=filters, kernel_size=kernel_size, activation=activation, batch_norm=batch_norm, dropout=dropout)
        m_up   = module_up(   m_mid,                filters=filters, kernel_size=kernel_size, activation=activation, batch_norm=batch_norm, dropout=dropout)
        if options.skip:
            m_up = concatenate([model, m_up])
        return m_up

def get_unet(_depth=2, _filters=16, _kernel_size=(3,3), _activation='prelu', _final_layer='softmax', _batch_norm=True, _dropout=0.5, _num_classes=1):
    layer_in  = Input(shape=(_ny,_nx,1))
    layer_adj = Activation('linear')(layer_in)

    layer_adj = Conv2D(filters=_filters, kernel_size=(1,1), padding='same', activation=_activation)(layer_adj)
#    layer_adj = Conv2D(filters=_filters, kernel_size=_kernel_size, padding='same', activation=_activation)(layer_adj)

    layer_mid = module_mid(layer_adj, depth=_depth, filters=_filters, kernel_size=_kernel_size, activation=_activation, batch_norm=_batch_norm, dropout=_dropout)
    layer_out = Dense(_num_classes, activation=_final_layer, use_bias=True)(layer_mid)
    model = Model(inputs=layer_in, outputs=layer_out)
    return model





# dsc = 1 - dsc_as_l2
#def dsc(y_true, y_pred, smooth=0.00001):
#    intersection = 2.0* K.abs(y_true * y_pred) + smooth
#    sumunion = K.sum(K.square(y_true), axis=(1,2)) + K.sum(K.square(y_pred), axis=(1,2)) + smooth
#    return -K.sum( intersection / K.expand_dims(K.expand_dims(sumunion, axis=1),axis=2), axis=(1,2))
def dsc(y_true, y_pred, smooth=0.00001):
    intersection = 2.0* K.sum( K.abs(y_true*y_pred), axis=(1,2)) + smooth
    union        = K.sum(K.square(y_true), axis=(1,2)) + K.sum(K.square(y_pred), axis=(1,2)) + smooth
    return -intersection/union
def dice_background(y_true, y_pred):
    dice = dsc(y_true, y_pred)
    return  -dice[:,0]
def dice_liver(y_true, y_pred):
    dice = dsc(y_true, y_pred)
    return  -dice[:,1]
def dice_tumor(y_true, y_pred):
    dice = dsc(y_true, y_pred)
    return  -dice[:,2]

# input x is a numpy array
# input a,b are scalars
def hard_sigmoid(a,b,x):
    lower = x < a 
    upper = x > b
    x[lower] = a
    x[upper] = b
    return x

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
     ('truthdata','(%d,%d)uint8' % (options.trainingresample,options.trainingresample)) ]

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
  print("idfold : ", idfold)
  print("slices total      : ", totnslice)
  print("slices training   : ", slicesplit)
  print("slices validation : ", totnslice - slicesplit)
  print("slices holdout    : ", len(numpydatabase[subsetidx_test]), "\n")


#  x_train = hard_sigmoid(-100.0, 150.0, x_train)


  # Convert to uint8 data and find out how many labels.
  y_train_typed = y_train.astype(np.uint8)
  t_max = np.max(y_train_typed)
  y_train_one_hot = to_categorical(y_train_typed, num_classes=t_max+1).reshape((y_train.shape)+(t_max+1,))
  print("Num classes : ", t_max+1)
  print("Shape before: {}; Shape after: {}".format(y_train.shape, y_train_one_hot.shape))
#  # Tumors are also in liver
#  liver = np.max(y_train_one_hot[...,1:], axis=3)
#  y_train_one_hot[...,1] = liver


  ###
  ### set up output, logging, and callbacks
  ###

  logfileoutputdir= '%s/%03d/%03d' % (options.outdir, kfolds, idfold)
  os.system ('mkdir -p %s' % logfileoutputdir)
  print("Output to\t", logfileoutputdir)

  if options.with_hvd:
      callbacks = [ hvd.callbacks.BroadcastGlobalVariablesCallback(0),
                    hvd.callbacks.MetricAverageCallback(),
                    hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=1),
                    keras.callbacks.TerminateOnNaN()         ]
      if hvd.rank() == 0:
          callbacks += [ keras.callbacks.ModelCheckpoint(filepath=logfileoutputdir+"/tumormodelunet.h5", verbose=1, save_best_only=True),
                         keras.callbacks.TensorBoard(log_dir=logfileoutputdir, histogram_freq=0, write_graph=True, write_images=False)      ]
  else:
      callbacks = [ keras.callbacks.TerminateOnNaN(),
                    keras.callbacks.ModelCheckpoint(filepath=logfileoutputdir+"/tumormodelunet.h5", verbose=1, save_best_only=True),  
                    keras.callbacks.TensorBoard(log_dir=logfileoutputdir, histogram_freq=0, write_graph=True, write_images=False)  ] 


  ###
  ### create and run model
  ###


  # set up optimizer
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


  # build model
  model = get_unet(_depth=options.depth, _filters=options.filters, _activation=options.activation, _num_classes=t_max+1, _batch_norm=options.batchnorm)
#  model.compile(loss="categorical_crossentropy",
  model.compile(loss=dsc,
        metrics=[dice_background, dice_liver, dice_tumor],
        optimizer=opt)
  print("Model parameters: {0:,}".format(model.count_params()))
  print("Input image shape: ", x_train[TRAINING_SLICES,:,:,np.newaxis].shape)

  model.summary()
 
#  train_gen = ImageDataGenerator()
#  valid_gen = ImageDataGenerator()
#  steps_per_epoch = len(x_train) // options.trainingbatch
#  train_iter = train_gen.flow(x_train[TRAINING_SLICES,...,np.newaxis],
#                              y_train_one_hot[TRAINING_SLICES],
#                              batch_size = options.trainingbatch)
#  valid_iter = valid_gen.flow(x_train[VALIDATION_SLICES,...,np.newaxis],
#                              y_train_one_hot[VALIDATION_SLICES],
#                              batch_size = options.validationbatch)
#
#  statevars = {'epoch':0, 'valloss':np.inf, 'lr':1.}
#
#  history = model.fit_generator(train_iter,
#                        steps_per_epoch=steps_per_epoch,
#                        validation_data = valid_iter,
#                        callbacks =callbacks, 
#                        verbose=1,
#                        initial_epoch=statevars['epoch'],
#                        epochs=options.numepochs)

  history = model.fit( x_train[TRAINING_SLICES,:,:,np.newaxis], 
                       y_train_one_hot[TRAINING_SLICES ],
                       validation_data=(x_train[VALIDATION_SLICES,:,:,np.newaxis], y_train_one_hot[VALIDATION_SLICES]),
                       callbacks = callbacks,
                       batch_size=options.trainingbatch,
                       epochs=options.numepochs)
 


  ###
  ### make predicions on validation set
  ###

  validationimgnii     = nib.Nifti1Image(x_train[VALIDATION_SLICES,:,:] , None )
  validationonehotnii  = nib.Nifti1Image(y_train[VALIDATION_SLICES,:,:] , None )
  y_predicted          = model.predict(  x_train[VALIDATION_SLICES,:,:,np.newaxis] )
  y_segmentation       = np.argmax(y_predicted , axis=-1)
  validationoutput     = nib.Nifti1Image( y_segmentation[:,:,:].astype(np.uint8), None )
  for jjj in range(t_max+1): 
      validationprediction = nib.Nifti1Image( y_predicted [:,:,:,jjj] , None )
      validationprediction.to_filename( logfileoutputdir+'/validationpredict-'+str(jjj)+'.nii.gz')
  validationimgnii.to_filename(     '%s/validationimg.nii.gz'     % logfileoutputdir )
  validationonehotnii.to_filename(  '%s/validationseg.nii.gz'     % logfileoutputdir )
  validationoutput.to_filename(     '%s/validationoutput.nii.gz'  % logfileoutputdir )

  modelloc = "%s/tumormodelunet.h5" % logfileoutputdir
  return modelloc


##########################
# apply model to test set
##########################
def MakeStatsScript(kfolds=options.kfolds, idfold=0):
  databaseinfo = GetDataDictionary()
  maketargetlist = []
  # open makefile
  with open('kfold%03d-%03d-stats.makefile' % (kfolds, idfold), 'w') as fileHandle:
      (train_set,test_set) = GetSetupKfolds(kfolds, idfold)
      for idtest in test_set:
         uidoutputdir= '%s/%03d/%03d' % (options.outdir, kfolds, idfold)
         segmaketarget  = '%s/label-%04d.nii.gz' % (uidoutputdir,idtest)
         segmaketarget0 = '%s/label-%04d-0.nii.gz' % (uidoutputdir,idtest)
         segmaketargetQ = '%s/label-%04d-?.nii.gz' % (uidoutputdir,idtest)
         predicttarget  = '%s/label-%04d-all.nii.gz' % (uidoutputdir,idtest)
         statstarget    = '%s/stats-%04d.txt' % (uidoutputdir,idtest)
         maketargetlist.append(segmaketarget )
         imageprereq = '$(TRAININGROOT)/%s' % databaseinfo[idtest]['image']
         segprereq   = '$(TRAININGROOT)/%s' % databaseinfo[idtest]['label']
         votecmd = "c3d %s -vote -type uchar -o %s" % (segmaketargetQ, predicttarget)
         infocmd = "c3d %s -info > %s" % (segmaketarget0,statstarget)
         statcmd = "c3d -verbose %s %s -overlap 0 -overlap 1 > %s" % (predicttarget, segprereq, statstarget)
         fileHandle.write('%s: %s\n' % (segmaketarget ,imageprereq ) )
         fileHandle.write('\t%s\n' % votecmd)
         fileHandle.write('\t%s\n' % infocmd)
         fileHandle.write('\t%s\n' % statcmd)
    # build job list
  with open('kfold%03d-%03d-stats.makefile' % (kfolds, idfold), 'r') as original: datastream = original.read()
  with open('kfold%03d-%03d-stats.makefile' % (kfolds, idfold), 'w') as modified: modified.write( 'TRAININGROOT=%s\n' % options.rootlocation + "cvtest: %s \n" % ' '.join(maketargetlist) + datastream)

##########################
# apply model to new data
##########################
def PredictModel(model=options.predictmodel, image=options.predictimage, outdir=options.segmentation):
  if (model != None and image != None and outdir != None ):
  
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    
    imagepredict = nib.load(image)
    imageheader  = imagepredict.header
    numpypredict = imagepredict.get_data().astype(IMG_DTYPE )
    # error check
    assert numpypredict.shape[0:2] == (_globalexpectedpixel,_globalexpectedpixel)
    nslice = numpypredict.shape[2]
    print(nslice)
    resizepredict = skimage.transform.resize(numpypredict,
            (options.trainingresample,options.trainingresample,nslice ),
            order=0,
            preserve_range=True,
            mode='constant').astype(IMG_DTYPE).transpose(2,1,0)


#    resizepredict = hard_sigmoid(-100.0, 150.0, resizepredict)


    # set up optimizer
    if options.with_hvd:
        if options.trainingsolver=="adam":
            opt = keras.optimizers.Adam(lr=0.001*hvd.size())
        elif options.trainingsolver=="adadelta":
            opt = keras.optimizers.Adadelta(1.0*hvd.size())
        elif options.trainingsolver=="nadam":
            opt = keras.optimizers.Nadam(0.002*hvd.size())
        elif options.trainingsolver=="sgd":
            opt = keras.optimizers.SGD(0.01*hvd*size())
        else:
            raise Exception("horovod-enabled optimizer not selected")
        opt = hvd.DistributedOptimizer(opt)
    else:
        opt = options.trainingsolver


    loaded_model = get_unet(_depth=options.depth, _filters=options.filters, _activation=options.activation, _num_classes=3, _batch_norm=options.batchnorm)
#    loaded_model.compile(loss="categorical_crossentropy",
    loaded_model.compile(loss=dsc,
          metrics=[dice_background, dice_liver, dice_tumor],
          optimizer=opt)
    loaded_model.load_weights(model)
    print("Loaded model from disk")

    segout = loaded_model.predict( resizepredict[...,np.newaxis] )
    for jjj in range(3):
        segout_resize = skimage.transform.resize(segout[...,jjj],
                (nslice,_globalexpectedpixel,_globalexpectedpixel),
                order=0,
                preserve_range=True,
                mode='constant').transpose(2,1,0)
        segout_img = nib.Nifti1Image(segout_resize, None, header=imageheader)
        segout_img.to_filename( outdir.replace('.nii.gz', '-%d.nii.gz' % jjj) )

################################
# Perform K-fold validation
################################
def OneKfold(k=options.kfolds, i=0, datadict=None):
    mdlloc = TrainModel(kfolds=k, idfold=i) 
    (train_set,test_set) = GetSetupKfolds(k,i)
    for idtest in test_set:
        baseloc = '%s/%03d/%03d' % (options.outdir, k, i)
        imgloc  = '%s/%s' % (options.rootlocation, datadict[idtest]['image'])
        outloc  = '%s/label-%04d.nii.gz' % (baseloc, idtest) 
        if options.numepochs > 0:
            PredictModel(model=mdlloc, image=imgloc, outdir=outloc )
    MakeStatsScript(kfolds=k, idfold=i)

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
