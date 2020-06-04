import numpy as np
import csv
import os
import json
import keras
from keras.layers import Input, Conv2D, LocallyConnected2D, Lambda, Add, Maximum, Minimum, Multiply, Dense, Layer, Activation, BatchNormalization
from keras.models import Model, Sequential
from keras.models import model_from_json, load_model
from keras.utils.np_utils import to_categorical
import keras.backend as K
from keras.callbacks import TensorBoard, TerminateOnNaN, ModelCheckpoint
from keras.callbacks import Callback as CallbackBase
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
                  action="store", dest="trainingsolver", default='adam',
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
parser.add_option("--nt",
                  type="int", dest="nt", default=10,
                  help="number of timesteps", metavar="int")
parser.add_option( "--randinit",
                  action="store_true", dest="randinit", default=False,
                  help="initialize u0 as random uniform. Default is constant 1", metavar="FILE")
parser.add_option( "--circleinit",
                  action="store_true", dest="circleinit", default=False,
                  help="initialize u0 as circle in lower right quadrant. Default is constant 1", metavar="FILE")
parser.add_option("--circlerad",
                  type="int", dest="circlerad", default=10,
                  help="radius of u0, when --circleinit is set", metavar="int")
parser.add_option( "--learned_edgekernel",
                  action="store_true", dest="learned_edgekernel", default=False,
                  help="use learned convolution kernels instead of FD stencils for transport", metavar="FILE")
parser.add_option( "--learned_kappakernel",
                  action="store_true", dest="learned_kappakernel", default=False,
                  help="use learned convolution kernels instead of FD stencils for curvature", metavar="FILE")
parser.add_option( "--alpha",
                  action="store_true", dest="alpha", default=False,
                  help="use alpha", metavar="FILE")
parser.add_option( "--beta",
                  action="store_true", dest="beta", default=False,
                  help="use beta", metavar="FILE")
parser.add_option( "--gamma",
                  action="store_true", dest="gamma", default=False,
                  help="use gamma", metavar="FILE")
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
_nt = options.nt
_nx = options.trainingresample
_ny = options.trainingresample
_num_classes = 2
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


  # create upwind FD kernels

def make_kernel(a):
    a = np.asarray(a)
    a = a.reshape(list(a.shape) + [1,1])
    return K.constant(a)

mXN = [[0,0,0],[-1,1,0],[0,0,0]]
mXP = [[0,0,0],[0,-1,1],[0,0,0]]
mYN = [[0,0,0],[0,1,0],[0,-1,0]]
mYP = [[0,1,0],[0,-1,0],[0,0,0]]
mXC = [[0,0,0],[-1,0,1],[0,0,0]]
mYC = [[0,1,0],[0,0,0],[0,-1,0]]
mLX = [[1,0,-1],[2,0,-2],[1,0,-1]]
mLY = [[1,2,1],[0,0,0],[-1,-2,-1]]
kXP = make_kernel(mXP) # first-order FD in x direction with prior information
kXN = make_kernel(mXN) # first-order FD in x direction with ahead information
kYP = make_kernel(mYP) # first-order FD in y direction with prior information
kYN = make_kernel(mYN) # first-order FD in y direction with ahead information
kXC = make_kernel(mXC) # second-order centered FD in x direction
kYC = make_kernel(mYC) # second-order centered FD in y direction
kLX = make_kernel(mLX) # Canny dx kernel
kLY = make_kernel(mLY) # Canny dy kernel
o   = make_kernel([[1]])


class ForcingFunction(Layer):

   def __init__(self, in_img, in_dims, **kwargs):
       self.img = in_img
       self.dt = 0.05
#       self.dims = in_dims
#       self.dx = in_dims[:,0,0,:]
#       self.dy = in_dims[:,1,0,:]
#       self.dz = in_dims[:,2,0,:]
#       self.shp = K.shape(in_img)
#       self.rhp = (self.shp[0], self.shp[1]*self.shp[2], self.shp[3])
       super(ForcingFunction, self).__init__(**kwargs)

   def build(self, input_shape):
       if options.alpha:
           self.alpha = self.add_weight(name='alpha',   # curvature coeff
                          shape=(1,1,1,1),
                          initializer='ones',
                          trainable=False)
       else:
           self.alpha = self.add_weight(name='alpha',   # curvature coeff
                          shape=(1,1,1,1),
                          initializer='zeros',
                          trainable=False)
       if options.beta:
           self.beta  = self.add_weight(name='beta',    # transport coeff
                          shape=(1,1,1,1),
                          initializer='ones',
                          trainable=False)
       else:
           self.beta  = self.add_weight(name='beta',    # transport coeff
                          shape=(1,1,1,1),
                          initializer='zeros',
                          trainable=False)
       if options.gamma:
           self.gamma = self.add_weight(name='gamma',   # balloon coeff
                          shape=(1,1,1,1),
                          initializer='ones',
                          trainable=False)
       else:
           self.gamma = self.add_weight(name='gamma',   # balloon coeff
                          shape=(1,1,1,1),
                          initializer='zeros',
                          trainable=False)
       if options.learned_edgekernel:
           self.kernel_edges = self.add_weight(name='edge_kernel',
                          shape=(3,3,1,16),
                          initializer='normal',
                          trainable=True)
       if options.learned_kappakernel:
           self.kernel_kappa_1 = self.add_weight(name='kappa_kernel_1',
                          shape=(3,3,1,16),
                          initializer='normal',
                          trainable=True)
           self.kernel_kappa_2 = self.add_weight(name='kappa_kernel_2',
                          shape=(3,3,16,48),
                          initializer='normal',
                          trainable=True)
       super(ForcingFunction, self).build(input_shape)
   def call(self, ins):
        u   = ins[0]
        img = ins[1]

        ### note : need to reshape before applying rescaling for each slice due to tf backend

        ### edge detection
        ### g_I(x) = 1 / (1 + norm(grad(I))^2 ), slightly modified

        if options.learned_edgekernel:
            edges = K.conv2d(self.img, self.kernel_edges, padding='same')
            edges = K.square(edges)
            edges = K.sum(edges, axis=-1, keepdims=True)
            edges = K.square(edges)
            edges = edges/K.max(edges)
            edges = K.constant(1.0) / (edges + K.constant(1.0))
        else:
            edges_x = K.conv2d(self.img, kLX, padding='same')
#            edges_x = K.reshape(edges_x, self.rhp)
#            edges_x = K.batch_dot(edges_x , self.dx, axes=[2,1])
#            edges_x = K.reshape(edges_x, self.shp)
            edges_y = K.conv2d(self.img, kLY, padding='same')
#            edges_y = K.reshape(edges_y, self.rhp)
#            edges_y = K.batch_dot(edges_y , self.dy, axes=[2,1])
#            edges_y = K.reshape(edges_y, self.shp)
            edges = K.square(edges_x) + K.square(edges_y)
            edges = K.square(edges)
            edges = edges/K.max(edges)
            edges = K.constant(1.0) / (20.0*edges + K.constant(1.0))



        ### grad( edge_detection )
        grad_edges_x = K.conv2d(edges, kXC, padding='same')
#        grad_edges_x = K.reshape(grad_edges_x, self.rhp)
#        grad_edges_x = K.batch_dot(grad_edges_x , self.dx, axes=[2,1])
#        grad_edges_x = K.reshape(grad_edges_x, self.shp)

        grad_edges_y = K.conv2d(edges, kYC, padding='same')
#        grad_edges_y = K.reshape(grad_edges_y, self.rhp)
#        grad_edges_y = K.batch_dot(grad_edges_y , self.dy, axes=[2,1])
#        grad_edges_y = K.reshape(grad_edges_y, self.shp)



	### transport - upwind approx to grad( edge_detection)^T grad( u )
        xp = K.conv2d(u, kXP, padding='same')
        xn = K.conv2d(u, kXN, padding='same')
        yp = K.conv2d(u, kYP, padding='same')
        yn = K.conv2d(u, kYN, padding='same')
        fxp =         K.relu(       grad_edges_x)
        fxn =  -1.0 * K.relu(-1.0 * grad_edges_x)
        fyp =         K.relu(       grad_edges_y)
        fyn =  -1.0 * K.relu(-1.0 * grad_edges_y)
        xpp = fxp*xp
        xnn = fxn*xn
        ypp = fyp*yp
        ynn = fyn*yn
        xterms = xpp + xnn
#        xterms = K.reshape(xterms, self.rhp)
#        xterms = K.batch_dot( xterms, self.dx, axes=[2,1])
#        xterms = K.reshape(xterms, self.shp)
        yterms = ypp + ynn
#        yterms = K.reshape(yterms, self.rhp)
#        yterms = K.batch_dot( yterms, self.dy, axes=[2,1])
#        yterms = K.reshape(yterms, self.shp)
        transport = 20.0*(xterms + yterms)



        ### curvature kappa( u )

        if options.learned_kappakernel:
             gradu = K.conv2d(u, self.kernel_kappa_1, padding='same')
             normu = K.square(gradu)
             normu = K.sum(normu, axis=-1, keepdims=True)
             normu = K.sqrt(normu + K.epsilon())
             gradu = gradu / (normu + K.epsilon())
             gradu = K.conv2d(gradu, self.kernel_kappa_2, padding='same')
             kappa = K.sum(gradu)
        else:
            gradu_x = K.conv2d(u, kXC, padding='same')
#            gradu_x = K.reshape(gradu_x, self.rhp)
#            gradu_x = K.batch_dot(gradu_x , self.dx, axes=[2,1])
#            gradu_x = K.reshape(gradu_x, self.shp)
            gradu_y = K.conv2d(u, kYC, padding='same')
#            gradu_y = K.reshape(gradu_y, self.rhp)
#            gradu_y = K.batch_dot(gradu_y , self.dy, axes=[2,1])
#            gradu_y = K.reshape(gradu_y, self.shp)
            normu = K.square(gradu_x) + K.square(gradu_y)
            normu = K.sqrt(normu + K.epsilon())
            kappa_x = gradu_x / (normu + K.epsilon())
            kappa_x = K.conv2d(kappa_x, kXC, padding='same')
#            kappa_x = K.reshape(kappa_x, self.rhp)
#            kappa_x = K.batch_dot(kappa_x , self.dx, axes=[2,1])
#            kappa_x = K.reshape(kappa_x, self.shp)
            kappa_y = gradu_y / (normu + K.epsilon())
            kappa_y = K.conv2d(kappa_y, kYC, padding='same')
#            kappa_y = K.reshape(kappa_y, self.rhp)
#            kappa_y = K.batch_dot(kappa_y , self.dy, axes=[2,1])
#            kappa_y = K.reshape(kappa_y, self.shp)
            kappa = kappa_x + kappa_y

        curvature = edges*kappa*normu

        ### balloon force
        balloon = edges*normu


        return u + K.constant(self.dt)*(\
                                           curvature * self.alpha \
                                        +  transport * self.beta  \
                                        +  balloon   * self.gamma )
   def compute_output_shape(self, input_shape):
       return input_shape



def get_upwind_transport_net(_nt, _final_sigma='sigmoid'):

    in_imgs   = Input(shape=(_ny,_nx,2))
    in_dims   = Input(shape=(3,1,1))      # provides structure dimension size for dx/dy/dz scaling

    in_img    = Lambda(lambda x : x[...,0][...,None])(in_imgs)   # I
#    img       = Conv2D(1, (1,1), padding='same', use_bias=True)(in_img)
#    img       = Activation('hard_sigmoid')(img)
    img = Activation('linear')(in_img)

    in_layer  = Lambda(lambda x : x[...,1][...,None])(in_imgs)   # u0
    mid_layer = Activation('linear')(in_layer)

    # Forcing Function F depends on image and on  u, but not on time
    F = ForcingFunction(img, in_dims)
    for ttt in range(_nt):
        mid_layer = F([mid_layer, img])

#    out_layer = Conv2D(1, (1,1), padding='same', use_bias=True)(mid_layer)
#    out_layer = Activation(_final_sigma)(out_layer)
    out_layer = Activation('linear')(mid_layer)
    model = Model([in_imgs, in_dims], out_layer)
    return model




# dsc = 1 - dsc_as_l2
def dsc_default(y_true, y_pred, smooth=0.00001):
    intersection = 2.0* K.abs(y_true * y_pred) + smooth
    sumunion = K.sum(K.square(y_true)) + K.sum(K.square(y_pred)) + smooth
    return -K.sum( intersection / K.expand_dims(K.expand_dims(K.expand_dims(sumunion, axis=0),axis=1),axis=2), axis=(0,1,2))
def dsc_as_l2(y_true, y_pred, smooth=0.00001):
    numerator = K.sum(K.square(y_true[...] - y_pred[...]),axis=(1,2)) + smooth
    denominator = K.sum(K.square(y_true[...]),axis=(1,2)) + K.sum(K.square(y_pred[...]),axis=(1,2)) + smooth
    disc = numerator/denominator
    return disc # average of dsc0,dsc1 over batch/stack
def dice_metric_zero(y_true, y_pred):
    batchdiceloss =  dsc_as_l2(y_true, y_pred)
    return 1.0 - batchdiceloss[:,0]




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
     ('truthdata','(%d,%d)uint8' % (options.trainingresample,options.trainingresample)),
     ('image_dx', float),
     ('image_dy', float),
     ('image_dz', float)     ]

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
        datamatrix ['image_dx']                       = np.repeat( 1.0/( float(imagedata.header['pixdim'][1])*(_globalexpectedpixel / options.trainingresample)), nslice)
        datamatrix ['image_dy']                       = np.repeat( 1.0/( float(imagedata.header['pixdim'][2])*(_globalexpectedpixel / options.trainingresample)), nslice)
        datamatrix ['image_dz']                       = np.repeat( 1.0/( float(imagedata.header['pixdim'][3])), nslice)
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

  global _num_classes

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
#  if np.sum(subsetidx_train) + np.sum(subsetidx_test) != min(np.sum(axialbounds ),np.sum(dbtrainindex )) :
#      raise("data error: slice numbers dont match")

  print('copy memory map from disk to RAM...')
  trainingsubset = numpydatabase[subsetidx_train]

  np.random.seed(seed=0)
  np.random.shuffle(trainingsubset)

  totnslice = len(trainingsubset)

  x_train=trainingsubset['imagedata']
  y_train=trainingsubset['truthdata']

  x_train_dx = trainingsubset['image_dx']
  x_train_dy = trainingsubset['image_dy']
  x_train_dz = trainingsubset['image_dz']
  x_train_dims = np.transpose(np.vstack((x_train_dx, x_train_dy, x_train_dz)))

  slicesplit        = int(0.9 * totnslice)

  TRAINING_SLICES   = slice(         0, slicesplit)
  VALIDATION_SLICES = slice(slicesplit, totnslice )

  print("\nkfolds : ", kfolds)
  print("idfold : ", idfold)
  print("slices total      : ", totnslice)
  print("slices training   : ", slicesplit)
  print("slices validation : ", totnslice - slicesplit)
  print("slices holdout    : ", len(numpydatabase[subsetidx_test]), "\n")

  # Convert to uint8 data and find out how many labels.
  y_train_typed = y_train.astype(np.uint8)
  _num_classes = 1
  y_train_one_hot = y_train_typed[...,np.newaxis]
  y_train_one_hot[ y_train_one_hot > 0 ] = 1  # map all nonzero values to 1

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

  # initial values for u
  if options.randinit:
      x_init = np.random.uniform(size=(totnslice,_nx,_ny))
  elif options.circleinit:
      x_center = int(_ny*0.25)
      y_center = int(_nx*0.67)
      rad      = options.circlerad
      x_init = np.zeros((totnslice,_nx,_ny))
#      x_init[:, y_center-rad:y_center+rad, x_center-rad:x_center+rad] = 1.0
      for xxx in range(2*rad):
           xloc = x_center - rad + xxx
           for yyy in range(2*rad):
               yloc = y_center - rad + yyy
               if (xxx - rad)**2 + (yyy - rad)**2 <= rad**2:
                   x_init[:, yloc, xloc] = 1.0
  else:
      x_init = np.ones((totnslice,_nx,_ny))

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

  # compile model graph
  model = get_upwind_transport_net(_nt)
  model.compile(loss=dsc_default,
#        metrics=[dice_metric_zero],
        optimizer=opt)
  print("Model parameters: {0:,}".format(model.count_params()))
  print("Input image shape: ", x_train[TRAINING_SLICES,:,:,np.newaxis].shape)
  x_in  = np.concatenate((x_train[TRAINING_SLICES  , :,:,np.newaxis], x_init[TRAINING_SLICES,   :,:,np.newaxis]), axis=-1)
  x_val = np.concatenate((x_train[VALIDATION_SLICES, :,:,np.newaxis], x_init[VALIDATION_SLICES, :,:,np.newaxis]), axis=-1)

  history = model.fit([    x_in,
                           x_train_dims[TRAINING_SLICES,:,np.newaxis,np.newaxis] ],
                       y_train_one_hot[TRAINING_SLICES ],
                       validation_data=([x_val, x_train_dims[VALIDATION_SLICES,:,np.newaxis,np.newaxis] ], y_train_one_hot[VALIDATION_SLICES]),
                       callbacks = callbacks,
                       batch_size=options.trainingbatch,
                       epochs=options.numepochs)

  modelWeights = []
  for layer in model.layers:
      layerWeights = []
      for weight in layer.get_weights():
          layerWeights.append(weight)
      modelWeights.append(layerWeights)
      print(layerWeights)

  ###
  ### make predicions on validation set
  ###

  validationimgnii     = nib.Nifti1Image(x_train[VALIDATION_SLICES,:,:] , None )
  validationonehotnii  = nib.Nifti1Image(y_train[VALIDATION_SLICES,:,:] , None )
  y_predicted          = model.predict( [x_val, x_train_dims[VALIDATION_SLICES,:,np.newaxis,np.newaxis] ])
  y_segmentation       = np.argmax(y_predicted , axis=-1)
  validationoutput     = nib.Nifti1Image( y_segmentation[:,:,:].astype(np.uint8), None )
  validationprediction  = nib.Nifti1Image(y_predicted [:,:,:,0] , None )
  validationprediction.to_filename( '%s/validationpredict.nii.gz' % logfileoutputdir )
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

    # init conditions
    inshape = resizepredict.shape
    if options.randinit:
        inits = np.random.uniform(size=inshape)
    elif options.circleinit:
        x_center = int(inshape[2]*0.25)
        y_center = int(inshape[1]*0.67)
        rad      = options.circlerad
        inits = np.zeros(inshape)
#        inits[:, y_center-rad:y_center+rad, x_center-rad:x_center+rad] = 1.0
        for xxx in range(2*rad):
             xloc = x_center - rad + xxx
             for yyy in range(2*rad):
                 yloc = y_center - rad + yyy
                 if (xxx - rad)**2 + (yyy - rad)**2 <= rad**2:
                     inits[:, yloc, xloc] = 1.0
    else:
        inits = np.ones(inshape)

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

    loaded_model = get_upwind_transport_net(_nt)
    loaded_model.compile(loss=dsc_default,
#          metrics=[dice_metric_zero],
          optimizer=opt)
    loaded_model.load_weights(model)
    print("Loaded model from disk")

    x_in = np.concatenate((resizepredict[...,np.newaxis], inits[...,np.newaxis]), axis=-1)
    x_dx = np.repeat( float(imageheader['pixdim'][1])*(_globalexpectedpixel / options.trainingresample), nslice)
    x_dy = np.repeat( float(imageheader['pixdim'][2])*(_globalexpectedpixel / options.trainingresample), nslice)
    x_dz = np.repeat( float(imageheader['pixdim'][3]), nslice)
    x_dims = np.transpose(np.vstack((x_dx, x_dy, x_dz)))

    segout = loaded_model.predict([x_in, x_dims[...,np.newaxis,np.newaxis]] )
    segout_resize = skimage.transform.resize(segout[...,0],
            (nslice,_globalexpectedpixel,_globalexpectedpixel),
            order=0,
            preserve_range=True,
            mode='constant').transpose(2,1,0)
    segout_img = nib.Nifti1Image(segout_resize, None, header=imageheader)
    segout_img.to_filename( outdir.replace('.nii.gz', '-%d.nii.gz' % 0) )

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
