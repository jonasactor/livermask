import numpy as np

# raw dicom data is usually short int (2bytes) datatype
# labels are usually uchar (1byte)
IMG_DTYPE = np.int16
SEG_DTYPE = np.uint8

# setup command line parser to control execution
from optparse import OptionParser
parser = OptionParser()
parser.add_option( "--builddb",
                  action="store_true", dest="builddb", default=False,
                  help="load all training data into npy", metavar="FILE")
parser.add_option( "--trainmodel",
                  action="store_true", dest="trainmodel", default=False,
                  help="train model", metavar="FILE")
parser.add_option( "--setuptestset",
                  action="store_true", dest="setuptestset", default=False,
                  help="cross validate test set", metavar="FILE")
parser.add_option( "--predictmodel",
                  action="store", dest="predictmodel", default=None,
                  help="apply model to image", metavar="Path")
parser.add_option( "--predictimage",
                  action="store", dest="predictimage", default=None,
                  help="apply model to image", metavar="Path")
parser.add_option( "--segmentation",
                  action="store", dest="segmentation", default=None,
                  help="model output ", metavar="Path")
parser.add_option( "--trainingsolver",
                  action="store", dest="trainingsolver", default='adadelta',
                  help="setup info", metavar="string")
parser.add_option( "--dbfile",
                  action="store", dest="dbfile", default="./trainingdata.csv",
                  help="training data file", metavar="string")
parser.add_option( "--trainingresample",
                  type="int", dest="trainingresample", default=256,
                  help="setup info", metavar="int")
parser.add_option( "--trainingbatch",
                  type="int", dest="trainingbatch", default=4,
                  help="setup info", metavar="int")
parser.add_option( "--kfolds",
                  type="int", dest="kfolds", default=5,
                  help="setup info", metavar="int")
parser.add_option( "--idfold",
                  type="int", dest="idfold", default=0,
                  help="setup info", metavar="int")
parser.add_option( "--rootlocation",
                  action="store", dest="rootlocation", default='/rsrch1/ip/jacctor/LiTS/LiTS',
                  help="setup info", metavar="string")
parser.add_option("--numepochs",
                  type="int", dest="numepochs", default=10,
                  help="number of epochs for training", metavar="int")
parser.add_option("--outdir",
                  action="store", dest="outdir", default='./',
                  help="directory for output", metavar="string")
parser.add_option("--nt",
                  type="int", dest="nt", default=10,
                  help="number of timesteps", metavar="int")
(options, args) = parser.parse_args()


# FIXME:  @jonasactor - is there a better software/programming practice to keep track  of the global variables?
_globalnpfile = options.dbfile.replace('.csv','%d.npy' % options.trainingresample )
_globalexpectedpixel=512
print('database file: %s ' % _globalnpfile )


# build data base from CSV file
def GetDataDictionary():
  import csv
  CSVDictionary = {}
  with open(options.dbfile, 'r') as csvfile:
    myreader = csv.DictReader(csvfile, delimiter=',')
    for row in myreader:
       CSVDictionary[int( row['dataid'])]  =  {'image':row['image'], 'label':row['label']}
  return CSVDictionary


# setup kfolds
def GetSetupKfolds(numfolds,idfold):
  import csv
  from sklearn.model_selection import KFold
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
     allkfolds = [ (train_index, test_index) for train_index, test_index in kf.split(dataidsfull )]
     train_index = allkfolds[idfold][0]
     test_index  = allkfolds[idfold][1]
  else:
     train_index = np.array(dataidsfull )
     test_index  = None
  print(numfolds, idfold)
  print("train_index:\t", train_index)
  print("test_index:\t", test_index)
  return (train_index,test_index)

##########################
# preprocess database and store to disk
##########################
if (options.builddb):
  import csv
  import nibabel as nib
  from scipy import ndimage
  import skimage.transform

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
        (liverboundingbox,tumorboundingbox                      )  = ndimage.find_objects(restruth)

      # error check
      if( nslice  == restruth.shape[2]):
        # custom data type to subset
        datamatrix = np.zeros(nslice  , dtype=mydatabasetype )

        # custom data type to subset
        datamatrix ['dataid']          = np.repeat(row['dataid']    ,nslice  )
        # id the slices within the bounding box
        axialliverbounds                              = np.repeat(False,nslice  )
        axialtumorbounds                              = np.repeat(False,nslice  )
        axialliverbounds[liverboundingbox[2]]         = True
        if (tumorboundingbox != None):
          axialtumorbounds[tumorboundingbox[2]]       = True
        datamatrix ['axialliverbounds'   ]            = axialliverbounds
        datamatrix ['axialtumorbounds'  ]             = axialtumorbounds
        datamatrix ['imagepath']                      = np.repeat(imagelocation ,nslice  )
        datamatrix ['truthpath']                      = np.repeat(truthlocation ,nslice  )
        datamatrix ['imagedata']                      = resimage.transpose(2,1,0)
        datamatrix ['truthdata']                      = restruth.transpose(2,1,0)
        numpydatabase = np.hstack((numpydatabase,datamatrix))
        # count total slice for QA
        totalnslice = totalnslice + nslice
      else:
        print('training data error image[2] = %d , truth[2] = %d ' % (nslice,restruth.shape[2]))

  # save numpy array to disk
  np.save( _globalnpfile,numpydatabase )


##########################
# build NN model from anonymized data
##########################
elif (options.trainmodel ):


  ###
  ### set up data
  ###

  # load database
  print('loading memory map db for large dataset')
  numpydatabase = np.load(_globalnpfile)

  #setup kfolds
  (train_index,test_index) = GetSetupKfolds(options.kfolds,options.idfold)

  print('copy data subsets into memory...')
  axialbounds = numpydatabase['axialliverbounds']
  dataidarray = numpydatabase['dataid']
  dbtrainindex= np.isin(dataidarray, train_index )
  dbtestindex = np.isin(dataidarray, test_index  )
  subsetidx_train  = np.all( np.vstack((axialbounds , dbtrainindex)) , axis=0 )
  subsetidx_test   = np.all( np.vstack((axialbounds , dbtestindex )) , axis=0 )

  if  np.sum(subsetidx_train   ) + np.sum(subsetidx_test) != min(np.sum(axialbounds ),np.sum(dbtrainindex )) :
    raise("data error")

  print('copy memory map from disk to RAM...')
  trainingsubset = numpydatabase[subsetidx_train   ]

  np.random.seed(seed=0)
  np.random.shuffle(trainingsubset )

  # subset within bounding box that has liver
  totnslice = len(trainingsubset)
  print("nslice train ",totnslice )

  # load training data as views
  x_train=trainingsubset['imagedata']
  y_train=trainingsubset['truthdata']
  slicesplit =  int(0.9 * totnslice )
  TRAINING_SLICES      = slice(0,slicesplit)
  VALIDATION_SLICES    = slice(slicesplit,totnslice)

  # Convert the labels into a one-hot representation
  from keras.utils.np_utils import to_categorical
  # Convert to uint8 data and find out how many labels.
  t=y_train.astype(np.uint8)
  t_max=np.max(t)
  print("Range of values: [0, {}]".format(t_max))
  y_train_one_hot = to_categorical(t, num_classes=t_max+1).reshape((y_train.shape)+(t_max+1,))
  print("Shape before: {}; Shape after: {}".format(y_train.shape, y_train_one_hot.shape))
  # The liver neuron should also be active for lesions within the liver
  liver = np.max(y_train_one_hot[:,:,:,1:], axis=3)
  y_train_one_hot[:,:,:,1]=liver



  ###
  ### set up NN
  ###

  _nt = options.nt
  _nx = options.trainingresample
  _ny = options.trainingresample
  from keras.layers import Input, Conv2D, LocallyConnected2D, Lambda, Add, Maximum, Minimum, Multiply, Dense, Layer, Activation, BatchNormalization
  from keras.models import Model, Sequential
  import keras.backend as K

 
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

  class Weight(Layer):

      def __init__(self, in_img, **kwargs):
          self.image = in_img
          super(Weight, self).__init__(**kwargs)

      def build(self, input_shape):
          self.kernel = self.add_weight(name='kernel',
                  shape=input_shape[1:],
                  initializer='normal',
                  trainable=True)
          super(Weight, self).build(input_shape)

      def call(self, x):
          return x - x + self.kernel

      def compute_output_shape(self, input_shape):
          return input_shape

  class ForcingFunction(Layer):

     def __init__(self, in_img, **kwargs):
         self.image = in_img
         super(ForcingFunction, self).__init__(**kwargs)

     def build(self, input_shape):
         self.conv_kernel_x = self.add_weight(name='conv_kernel_x',
                          shape=(3,3,1,16),
                          initializer='normal',
                          trainable=True)
         self.conv_kernel_y = self.add_weight(name='conv_kernel_y',
                          shape=(3,3,1,16),
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
          kappa = K.sum(kappa, axis=-1, keepdims=True)

#          uxx = K.conv2d(u, kXX, padding='same')
#          uyy = K.conv2d(u, kYY, padding='same')
#          uxy = K.conv2d(u, kXY, padding='same')
#          uxc = K.conv2d(u, kXC, padding='same')
#          uyc = K.conv2d(u, kYC, padding='same')
#          kappa = uxx*(uyc*uyc) - 2.0*(uxc*uyc)*uxy + uyy*(uxc*uxc)

          return u + xpp + xnn + ypp + ynn + edges + kappa

     def compute_output_shape(self, input_shape):
          return input_shape

  def get_upwind_transport_net(_nt, _final_sigma='relu', _num_classes=1):
      in_img  = Input(shape=(_ny,_nx,1))

      # initialization for u_0
      in_init   = Input(shape=(_ny,_nx,1))
      mid_layer = Activation('relu')(in_init)
#      mid_layer = Conv2D(1, (3,3), padding='same')(in_init)
#      mid_layer = Activation('relu')(mid_layer)
#      mid_layer = Conv2D(1, (3,3), padding='same')(mid_layer)
#      mid_layer = Activation('relu')(mid_layer)

      # Forcing Function F depends on image and on  u, but not on time
      F = ForcingFunction(in_img)

      for ttt in range(_nt):
          mid_layer = F(mid_layer)      
  
      out_layer = Conv2D(_num_classes, (1,1), padding='same')(mid_layer)
      out_layer = Activation('relu')(out_layer)
      model = Model([in_img, in_init], out_layer)
      return model



  ###
  ### set up Dice scores
  ###

  ### Train model with Dice loss
  def dsc(y_true, y_pred, smooth=0):
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
  def dsc_as_l2(y_true, y_pred):
      numerator = K.sum(K.square(y_true - y_pred),axis=(1,2))
      denominator = K.sum(K.square(y_true),axis=(1,2)) + K.sum(K.square(y_pred),axis=(1,2))
      dsc = numerator/denominator
      return dsc

  def dice_metric_zero(y_true, y_pred):
      batchdiceloss =  dsc(y_true, y_pred)
      return -batchdiceloss[:,0]
  def dice_metric_one(y_true, y_pred):
      batchdiceloss =  dsc(y_true, y_pred)
      return -batchdiceloss[:,1]
  def dice_metric_two(y_true, y_pred):
      batchdiceloss =  dsc(y_true, y_pred)
      return -batchdiceloss[:,2]



  ###
  ### set up output and logging
  ###

  # output location
  logfileoutputdir= '%s/%03d/%03d' % (options.outdir, options.kfolds, options.idfold)
  print(logfileoutputdir)
  import os
  os.system ('mkdir -p %s' % logfileoutputdir)

  from keras.callbacks import TensorBoard
  from keras.callbacks import Callback as CallbackBase

  tensorboard = TensorBoard(log_dir=logfileoutputdir, histogram_freq=0, write_graph=True, write_images=False)

  class MyHistories(CallbackBase):
      def on_train_begin(self, logs={}):
          self.min_valloss = np.inf

      def on_train_end(self, logs={}):
          return

      def on_epoch_begin(self, epoch, logs={}):
          return

      def on_epoch_end(self, epoch, logs={}):
          if logs.get('val_loss')< self.min_valloss :
             self.min_valloss = logs.get('val_loss')
             # https://machinelearningmastery.com/save-load-keras-deep-learning-models/
             # serialize model to JSON
             model_json = model.to_json()
             with open("%s/tumormodelunet.json" % logfileoutputdir , "w") as json_file:
                 json_file.write(model_json)
             # serialize weights to HDF5
             model.save_weights("%s/tumormodelunet.h5" % logfileoutputdir )
             print("Saved model to disk - val_loss", self.min_valloss  )
          return

      def on_batch_begin(self, batch, logs={}):
          return

      def on_batch_end(self, batch, logs={}):
          return
  callbacksave = MyHistories()



  ###
  ### create and run model
  ###

  x_init_train = np.fromfunction(lambda z,i,j : np.maximum(20. - ( (_ny/2.0-i)**2 + (_nx/2.0-j)**2 ), 0.0), (slicesplit,_ny,_nx))
  x_init_valid = np.fromfunction(lambda z,i,j : np.maximum(20. - ( (_ny/2.0-i)**2 + (_nx/2.0-j)**2 ), 0.0), (totnslice-slicesplit,_ny,_nx))

  model = get_upwind_transport_net(_nt, _final_sigma='sigmoid', _num_classes=t_max+1)
  model.compile(loss=dsc_as_l2,
        metrics=[dsc,dice_metric_zero,dice_metric_one,dice_metric_two],
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

  import nibabel as nib
  validationimgnii                = nib.Nifti1Image(x_train[VALIDATION_SLICES,:,:] , None )
  validationimgnii.to_filename( '%s/validationimg.nii.gz' % logfileoutputdir )
  validationonehotnii             = nib.Nifti1Image(y_train[VALIDATION_SLICES  ,:,:] , None )
  validationonehotnii.to_filename( '%s/validationseg.nii.gz' % logfileoutputdir )
  y_predicted                     = model.predict([x_train[VALIDATION_SLICES,:,:,np.newaxis], x_init_valid[:,:,:,np.newaxis]])
  y_segmentation                  = np.argmax(y_predicted , axis=-1)
  for jjj in range(3):
      validationprediction            = nib.Nifti1Image(y_predicted [:,:,:,jjj] , None )
      validationprediction.to_filename( '%s/validationpredict-%d.nii.gz' % (logfileoutputdir,jjj) )
  validationoutput                = nib.Nifti1Image( y_segmentation[:,:,:].astype(np.uint8), None )
  validationoutput.to_filename( '%s/validationoutput.nii.gz' % logfileoutputdir )



##########################
# apply model to test set
##########################
elif (options.setuptestset):
  databaseinfo = GetDataDictionary()

  maketargetlist = []
  # open makefile
  with open('kfold%03d-predict.makefile' % options.kfolds ,'w') as fileHandle:
    for iii in range(options.kfolds):
      (train_set,test_set) = GetSetupKfolds(options.kfolds,iii)
      for idtest in test_set:
         uidoutputdir= '%s/%03d/%03d' % (options.outdir, options.kfolds, iii)
         segmaketarget  = '%s/label-%04d.nii.gz' % (uidoutputdir,idtest)
         maketargetlist.append(segmaketarget )
         imageprereq = '$(TRAININGROOT)/%s' % databaseinfo[idtest]['image']
         cvtestcmd = "python3 ./liver2.py --predictimage=%s --predictmodel=%s/tumormodelunet.json --segmentation=%s --dbfile=%s"  % (imageprereq ,uidoutputdir,segmaketarget ,options.dbfile)
         fileHandle.write('%s: %s\n' % (segmaketarget ,imageprereq ) )
         fileHandle.write('\t%s\n' % cvtestcmd)
  # build job list
  with open('kfold%03d-predict.makefile' % options.kfolds, 'r') as original: datastream = original.read()
  with open('kfold%03d-predict.makefile' % options.kfolds, 'w') as modified: modified.write( 'TRAININGROOT=%s\n' % options.rootlocation + "cvtest: %s \n" % ' '.join(maketargetlist) + datastream)

  with open('kfold%03d-stats.makefile' % options.kfolds, 'w') as fileHandle:
    for iii in range(options.kfolds):
      (train_set,test_set) = GetSetupKfolds(options.kfolds,iii)
      for idtest in test_set:
         uidoutputdir= '%s/%03d/%03d' % (options.outdir, options.kfolds, iii)
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
         statcmd = "c3d -verbose %s %s -overlap 0 -overlap 1 -overlap 2 > %s" % (predicttarget, segprereq, statstarget)
         fileHandle.write('%s: %s\n' % (segmaketarget ,imageprereq ) )
         fileHandle.write('\t%s\n' % votecmd)
         fileHandle.write('\t%s\n' % infocmd)
         fileHandle.write('\t%s\n' % statcmd)
  # build job list
  with open('kfold%03d-stats.makefile' % options.kfolds, 'r') as original: datastream = original.read()
  with open('kfold%03d-stats.makefile' % options.kfolds, 'w') as modified: modified.write( 'TRAININGROOT=%s\n' % options.rootlocation + "cvtest: %s \n" % ' '.join(maketargetlist) + datastream)

##########################
# apply model to new data
##########################
elif (options.predictmodel != None and options.predictimage != None and options.segmentation != None ):
  import json
  import nibabel as nib
  import skimage.transform
  # force cpu for debug
  import os
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
  # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
  from keras.models import model_from_json
  # load json and create model
  _glexpx = _globalexpectedpixel
  json_file = open(options.predictmodel, 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  loaded_model = model_from_json(loaded_model_json)
  # load weights into new model
  weightsfile= '.'.join(options.predictmodel.split('.')[0:-1]) + '.h5'
  loaded_model.load_weights(weightsfile)
  print("Loaded model from disk")

  imagepredict = nib.load(options.predictimage)
  imageheader  = imagepredict.header
  numpypredict = imagepredict.get_data().astype(IMG_DTYPE )
  # error check
  assert numpypredict.shape[0:2] == (_glexpx,_glexpx)
  nslice = numpypredict.shape[2]
  print(nslice)
  resizepredict = skimage.transform.resize(numpypredict,
          (options.trainingresample,options.trainingresample,nslice ),
          order=0,
          preserve_range=True,
          mode='constant').astype(IMG_DTYPE).transpose(2,1,0)

  # FIXME: @jonasactor - the numlabel will change depending on the training data... can you make this more robust and the number of labels from the model?
  numlabel = 3

  segout = loaded_model.predict(resizepredict[:,:,:,np.newaxis] )
  for jjj in range(numlabel):
      segout_resize = skimage.transform.resize(segout[...,jjj],
              (nslice,_glexpx,_glexpx),
              order=0,
              preserve_range=True,
              mode='constant').transpose(2,1,0)
      segout_img = nib.Nifti1Image(segout_resize, None, header=imageheader)
      segout_img.to_filename( options.segmentation.replace('.nii.gz', '-%d.nii.gz' % jjj) )

#########################
# print help
#########################
else:
  parser.print_help()
