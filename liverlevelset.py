import numpy as np
import csv
import os
import json
from keras.layers import Input, Conv2D, LocallyConnected2D, Lambda, Add, Maximum, Minimum, Multiply, Dense, Layer, Activation, BatchNormalization
from keras.models import Model, Sequential
from keras.models import model_from_json, load_model
from keras.utils.np_utils import to_categorical
import keras.backend as K
from keras.callbacks import TensorBoard
from keras.callbacks import Callback as CallbackBase
from optparse import OptionParser
import nibabel as nib
from scipy import ndimage
from sklearn.model_selection import KFold
import skimage.transform


# setup command line parser to control execution
parser = OptionParser()
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
                  action="store", dest="trainingsolver", default='adadelta',
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
parser.add_option( "--trainallslices",
                  action="store_true", dest="trainallslices", default=False,
                  help="train using all slices, including those without liver", metavar="FILE")
(options, args) = parser.parse_args()

# raw dicom data is usually short int (2bytes) datatype
# labels are usually uchar (1byte)
IMG_DTYPE = np.int16
SEG_DTYPE = np.uint8

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
  print(numfolds, idfold)
  print("train_index:\t", train_index)
  print("test_index:\t",  test_index)
  return (train_index,test_index)

 
  # create upwind FD kernels
kXP = K.constant(np.asarray([[-1,1,0]])[:,:,np.newaxis,np.newaxis])
kXN = K.constant(np.asarray([[0,-1,1]])[:,:,np.newaxis,np.newaxis])
kYP = K.constant(np.asarray([[0],[1],[-1]])[:,:,np.newaxis,np.newaxis])
kYN = K.constant(np.asarray([[1],[-1],[0]])[:,:,np.newaxis,np.newaxis])
kXC = K.constant(np.asarray([[-1,0,1]])[:,:,np.newaxis,np.newaxis])
kYC = K.constant(np.asarray([[1],[0],[-1]])[:,:,np.newaxis,np.newaxis])
kXX = K.constant(np.asarray([[-1,2,1]])[:,:,np.newaxis,np.newaxis])
kYY = K.constant(np.asarray([[1],[2],[-1]])[:,:,np.newaxis,np.newaxis])
kXY = K.constant(np.asarray([[-1,0,1],[0,0,0],[1,0,-1]])[:,:,np.newaxis,np.newaxis])
blur = K.constant(np.asarray([[0.0625, 0.1250, 0.0625],[0.1250, 0.5000, 0.1250],[0.0625, 0.1250, 0.0625]])[:,:,np.newaxis,np.newaxis])


class ForcingFunction(Layer):

   def __init__(self, in_img, **kwargs):
       self.image = in_img
       super(ForcingFunction, self).__init__(**kwargs)

   def build(self, input_shape):
       self.conv_kernel_x = self.add_weight(name='conv_kernel_x_0',
                        shape=(3,3,1,32),
                        initializer='normal',
                        trainable=True)
       self.conv_kernel_y = self.add_weight(name='conv_kernel_y_0',
                        shape=(3,3,1,32),
                        initializer='normal',
                        trainable=True)
       self.sep_kernel_depthwise = self.add_weight(name='sep_kernel_depth',
                        shape=(3,3,1,8),
                        initializer='normal',
                        trainable=True)
       self.sep_kernel_pointwise = self.add_weight(name='sep_kernel_point',
                        shape=(1,1,8,8),
                        initializer='normal',
                        trainable=True)
       self.conv_kernel_curve_0 = self.add_weight(name='conv_kernel_curve_0',
                        shape=(5,5,1,8),
                        initializer='normal',
                        trainable=True)
       self.conv_kernel_curve_1 = self.add_weight(name='conv_kernel_curve_1',
                        shape=(5,5,8,8),
                        initializer='normal',
                        trainable=True)
       self.conv_kernel_curve_2 = self.add_weight(name='conv_kernel_curve_2',
                        shape=(3,3,8,16),
                        initializer='normal',
                        trainable=True)
       self.conv_kernel_smoother = self.add_weight(name='conv_kernel_smoother',
                        shape=(5,5,1,1),
                        initializer='normal',
                        trainable=True)
       self.kappa_kernel = self.add_weight(name='kappa_kernel',
                        shape=input_shape[1:],
                        initializer='normal',
                        trainable=True)
       super(ForcingFunction, self).build(input_shape)

   def call(self, u):
        
        # edge detection (learned filter)
        edges = K.separable_conv2d(self.image, self.sep_kernel_depthwise, self.sep_kernel_pointwise, padding='same')
        edges = K.relu(edges)
        edges = K.sum(edges, axis=-1, keepdims=True)
        edges = K.softsign(edges)
 
        # grad( edge_detection ) approx (learned filter)
        grad_edges_x = K.conv2d(edges, self.conv_kernel_x, padding='same')
        grad_edges_x = K.relu(grad_edges_x)
        grad_edges_x = K.sum(grad_edges_x, axis=-1, keepdims=True)
        grad_edges_y = K.conv2d(edges, self.conv_kernel_y, padding='same')
        grad_edges_y = K.relu(grad_edges_y)
        grad_edges_y = K.sum(grad_edges_y, axis=-1, keepdims=True)

        # upwind approx to grad( edge_detection)^T grad( u )
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

        # curvature kappa( u ) approx (learned filter)
        kappa = K.conv2d(u, self.conv_kernel_curve_0, padding='same')
        kappa = K.conv2d(kappa, self.conv_kernel_curve_1, padding='same')
        kappa = K.conv2d(kappa, self.conv_kernel_curve_2, padding='same')
        kappa = K.sum(kappa, axis=-1, keepdims=True)
        kappa = kappa*self.kappa_kernel
        kappa = K.conv2d(kappa, self.conv_kernel_smoother, padding='same')

        return u + xpp + xnn + ypp + ynn + edges + kappa

   def compute_output_shape(self, input_shape):
       return input_shape


def get_upwind_transport_net(_nt, _final_sigma='softmax'):
    global _num_classes

    in_img    = Input(shape=(_ny,_nx,1))
    in_layer  = Input(shape=(_ny,_nx,1))
    mid_layer = Conv2D(1, (5,5), padding='same')(in_layer)
    mid_layer = Conv2D(1, (5,5), padding='same')(mid_layer)
    mid_layer = Conv2D(1, (5,5), padding='same')(mid_layer)

    # Forcing Function F depends on image and on  u, but not on time
    F = ForcingFunction(in_img)
    for ttt in range(_nt):
        mid_layer = F(mid_layer)
  
    out_layer = Conv2D(_num_classes, (5,5), padding='same')(mid_layer)
    out_layer = Activation(_final_sigma)(out_layer)
    model = Model([in_img, in_layer], out_layer)
    return model



def dsc(y_true, y_pred, smooth=0.00001):
      """
      Dice = \sum_Nbatch \sum_Nonehot (2*|X & Y|)/ (|X|+ |Y|)
           = \sum_Nbatch \sum_Nonehot  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
      return negative dice value for minimization. one dsc per one hot image for each batch. Nbatch * Nonehot total images.
      objective function has implicit reduce mean -  /opt/apps/miniconda/miniconda3/lib/python3.6/site-packages/keras/engine/training.py(447)weighted()
      """
      # DSC = DSC_image1 +  DSC_image2 + DSC_image3 + ...
      intersection = 2. *K.abs(y_true * y_pred) + smooth
      sumunion = K.sum(K.square(y_true),axis=(1,2)) + K.sum(K.square(y_pred),axis=(1,2)) + smooth
      dicevalues= K.sum(intersection / K.expand_dims(K.expand_dims(sumunion,axis=1),axis=2), axis=(1,2))
      return -dicevalues

# dsc = 1 - dsc_as_l2
def dsc_as_l2(y_true, y_pred, smooth=0.00001):
    numerator = K.sum(K.square(y_true - y_pred),axis=(1,2)) + smooth
    denominator = K.sum(K.square(y_true),axis=(1,2)) + K.sum(K.square(y_pred),axis=(1,2)) + smooth
    disc = numerator/denominator
    return disc # average of dsc0,dsc1 over batch/stack 
def dice_metric_zero(y_true, y_pred):
    batchdiceloss =  dsc_as_l2(y_true, y_pred)
    return 1.0 - batchdiceloss[:,0]
def dice_metric_one(y_true, y_pred):
    batchdiceloss =  dsc_as_l2(y_true, y_pred)
    return 1.0 - batchdiceloss[:,1]

class MyHistories(CallbackBase):
    def __init__(self, outloc, **kwargs):
       self.outloc = outloc
       super(MyHistories, self).__init__(**kwargs)
   
    def on_train_begin(self, logs={}):
        self.min_valloss = np.inf

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        if logs.get('val_loss')< self.min_valloss :
           self.min_valloss = logs.get('val_loss')
           self.model.save("%s/tumormodelunet.h5" % self.outloc)
           print("Saved model to disk - val_loss", self.min_valloss  )
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


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
     ('truthpath','S128'),('truthdata','(%d,%d)uint8' % (options.trainingresample,options.trainingresample))]

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
  if options.trainallslices:
      subsetidx_train  = dbtrainindex
      subsetidx_test   = dbtestindex
  else:
    subsetidx_train  = np.all( np.vstack((axialbounds , dbtrainindex)) , axis=0 )
    subsetidx_test   = np.all( np.vstack((axialbounds , dbtestindex )) , axis=0 )
    if  np.sum(subsetidx_train) + np.sum(subsetidx_test) != min(np.sum(axialbounds ),np.sum(dbtrainindex )) :
      print(np.sum(subsetidx_train))
      print(np.sum(subsetidx_test))
      print(np.sum(axialbounds))
      print(np.sum(dbtrainindex))
      raise("data error: slice numbers dont match")

  print('copy memory map from disk to RAM...')
  trainingsubset = numpydatabase[subsetidx_train]

  np.random.seed(seed=0)
  np.random.shuffle(trainingsubset)

  totnslice = len(trainingsubset)
  print("nslice train ", totnslice)

  x_train=trainingsubset['imagedata']
  y_train=trainingsubset['truthdata']
  slicesplit        =   int(0.9 * totnslice)
  TRAINING_SLICES   = slice(         0, slicesplit)
  VALIDATION_SLICES = slice(slicesplit, totnslice )

  # Convert the labels into a one-hot representation
  # Convert to uint8 data and find out how many labels.
  y_train_typed = y_train.astype(np.uint8)
  _num_classes = np.max( y_train_typed )+1
  print("Range of values: [0, {}]".format(_num_classes-1))

  y_train_one_hot_all = to_categorical(y_train_typed, num_classes=_num_classes).reshape((y_train.shape)+(_num_classes,))
  # The liver neuron should also be active for lesions within the liver
  y_train_one_hot_all[:,:,:,1] = np.max(y_train_one_hot_all[:,:,:,1:], axis=3)
  y_train_one_hot = y_train_one_hot_all[:,:,:,:2]
  _num_classes=2


  ###
  ### set up output and logging
  ###

  logfileoutputdir= '%s/%03d/%03d' % (options.outdir, kfolds, idfold)
  os.system ('mkdir -p %s' % logfileoutputdir)
  print(logfileoutputdir)

  tensorboard = TensorBoard(log_dir=logfileoutputdir, histogram_freq=0, write_graph=True, write_images=False)
  callbacksave = MyHistories(outloc=logfileoutputdir)


  ###
  ### create and run model
  ###

  x_init_train = np.ones((slicesplit,_nx,_ny))
  x_init_valid = np.ones((totnslice-slicesplit,_nx,_ny))
#  x_init_train = np.random.uniform(size=(slicesplit,_nx,_ny))
#  x_init_valid = np.random.uniform(size=(totnslice-slicesplit,_nx,_ny))


  model = get_upwind_transport_net(_nt)
  model.compile(loss=dsc_as_l2,
        metrics=[dice_metric_zero,dice_metric_one],
        optimizer=options.trainingsolver)
  print("Model parameters: {0:,}".format(model.count_params()))
  print("Input shape: ", x_train[TRAINING_SLICES,:,:,np.newaxis].shape)
  history = model.fit([x_train[TRAINING_SLICES ,:,:,np.newaxis], x_init_train[:,:,:,np.newaxis]],
                          y_train_one_hot[TRAINING_SLICES ],
                          validation_data=([x_train[VALIDATION_SLICES,:,:,np.newaxis], x_init_valid[:,:,:,np.newaxis]], y_train_one_hot[VALIDATION_SLICES]),
                          callbacks = [tensorboard,callbacksave],
                          batch_size=options.trainingbatch,
                          epochs=options.numepochs)

  ###
  ### make predicions on validation set
  ###

  validationimgnii     = nib.Nifti1Image(x_train[VALIDATION_SLICES,:,:] , None )
  validationonehotnii  = nib.Nifti1Image(y_train[VALIDATION_SLICES,:,:] , None )
  y_predicted          = model.predict([x_train[VALIDATION_SLICES,:,:,np.newaxis], x_init_valid[:,:,:,np.newaxis]])
  y_segmentation       = np.argmax(y_predicted , axis=-1)
  validationoutput     = nib.Nifti1Image( y_segmentation[:,:,:].astype(np.uint8), None )
  for jjj in range(_num_classes):
      validationprediction  = nib.Nifti1Image(y_predicted [:,:,:,jjj] , None )
      validationprediction.to_filename( '%s/validationpredict-%d.nii.gz' % (logfileoutputdir,jjj) )
  validationimgnii.to_filename(    '%s/validationimg.nii.gz'    % logfileoutputdir )
  validationonehotnii.to_filename( '%s/validationseg.nii.gz'    % logfileoutputdir )
  validationoutput.to_filename(    '%s/validationoutput.nii.gz' % logfileoutputdir )

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
    inits = np.ones(inshape)
#    inits = np.random.uniform(size=inshape)

    loaded_model = get_upwind_transport_net(_nt)
    loaded_model.compile(loss=dsc_as_l2,
          metrics=[dice_metric_zero,dice_metric_one],
          optimizer=options.trainingsolver)
    loaded_model.load_weights(model)
    print("Loaded model from disk")

    segout = loaded_model.predict([resizepredict[:,:,:,np.newaxis], inits[:,:,:,np.newaxis]] )
    for jjj in range(_num_classes):
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
