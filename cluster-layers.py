import numpy as np
import os
import sys
import keras
from keras.layers import Input, Conv2D, UpSampling2D, Lambda, SpatialDropout2D, Dense, Layer, Activation, BatchNormalization, MaxPool2D, concatenate, LocallyConnected2D
from keras.models import Model, Sequential
from keras.models import model_from_json, load_model
from keras.utils import multi_gpu_model
from keras.utils.np_utils import to_categorical
import keras.backend as K
from keras.callbacks import TensorBoard, TerminateOnNaN, ModelCheckpoint
from optparse import OptionParser # TODO update to ArgParser (python2 --> python3)
import nibabel as nib
from scipy import ndimage
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
try:
    from sklearn.metrics import davies_bouldin_score as cluster_score
except:
    from sklearn.metrics import calinski_harabaz_score as cluster_score
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
parser.add_option( "--trainingsolver",
                  action="store", dest="trainingsolver", default='adam',
                  help="setup info", metavar="string")
parser.add_option( "--trainingresample",
                  type="int", dest="trainingresample", default=256,
                  help="resample so that model prediction occurs at this resolution", metavar="int")
parser.add_option( "--rootlocation",
                  action="store", dest="rootlocation", default='/rsrch1/ip/jacctor/LiTS/LiTS',
                  help="root location for images for training", metavar="Path")
parser.add_option("--outdir",
                  action="store", dest="outdir", default='./',
                  help="directory for output", metavar="Path")
parser.add_option( "--skip",
                  action="store_true", dest="skip", default=False,
                  help="skip connections in UNet", metavar="bool")
parser.add_option( "--fanout",
                  action="store_true", dest="fanout", default=False,
                  help="fan out as UNet gets deeper (more filters at deeper levels)", metavar="bool")
parser.add_option( "--batchnorm",
                  action="store_true", dest="batchnorm", default=False,
                  help="use batch normalization in UNet", metavar="bool")
parser.add_option( "--depth",
                  type="int", dest="depth", default=2,
                  help="number of down steps to UNet", metavar="int")
parser.add_option( "--filters",
                  type="int", dest="filters", default=16,
                  help="number of filters for output of CNN layer", metavar="int")
parser.add_option( "--clusters",
                  type="int", dest="clusters", default=5,
                  help="number of kmeans clusters", metavar="int")
parser.add_option( "--activation",
                  action="store", dest="activation", default='relu',
                  help="activation function", metavar="string")
parser.add_option( "--segthreshold",
                  type="float", dest="segthreshold", default=0.5,
                  help="cutoff for binary segmentation from real-valued output", metavar="float")
parser.add_option( "--predictlivermodel",
                  action="store", dest="predictlivermodel", default="/rsrch1/ip/jacctor/livermask/unet-2gpu-alldata-depth3/001/000/liver/modelunet.h5",
                  help="model weights (.h5) for liver seg prediction", metavar="Path")
parser.add_option( "--normalize",
                  action="store_true", dest="normalize", default=False,
                  help="normalize data matrix for clustering", metavar="bool")
parser.add_option( "--show_hist",
                  action="store_true", dest="show_hist", default=False,
                  help="show plotted histograms", metavar="bool")
parser.add_option( "--show_clusters",
                  action="store_true", dest="show_clusters", default=False,
                  help="show plotted clusters", metavar="bool")
parser.add_option( "--test_range_clusters",
                  action="store_true", dest="test_range_clusters", default=False,
                  help="try kmeans with various numbers of clusters", metavar="bool")
parser.add_option( "--tsne",
                  action="store_true", dest="tsne", default=False,
                  help="use TSNE for clustering", metavar="bool")
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


_globalexpectedpixel=512
_nx = options.trainingresample
_ny = options.trainingresample




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

def get_unet(_depth=2, _filters=16, _kernel_size=(3,3), _activation='prelu', _final_layer='sigmoid', _batch_norm=True, _dropout=0.5, _num_classes=1):
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

def hard_sigmoid(a,b,x):
    lower = x < a
    upper = x > b
    xx = np.copy(x)
    xx[lower] = a
    xx[upper] = b
    return xx

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


def load_model(livermodel=options.predictlivermodel):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    
    opt = GetOptimizer()
    loaded_liver_model = get_unet(_depth=options.depth, _filters=options.filters, _activation=options.activation, _num_classes=1, _batch_norm=options.batchnorm)
    loaded_liver_model.compile(loss=dsc_l2, optimizer=opt)
    loaded_liver_model.load_weights(livermodel)

    if options.gpu > 1:
       inp = loaded_liver_model.input
       layer_dict = dict([(layer.name, layer) for layer in loaded_liver_model.layers])
       model_dict = dict([(layer.name, layer) for layer in layer_dict['model_1'].layers])

       output_dict = dict([(layer.name, layer.output) for layer in layer_dict['model_1'].layers])
       functor_dict = dict([(layer.name, K.function([inp, K.learning_phase()], [layer.output])) for layer in layer_dict['model_1'].layers])

       return model_dict, output_dict, functor_dict
    else:
        raise Exception("1-gpu version not implemented yet")




def create_slice(loc=options.outdir):

        imagepredict = nib.load('/rsrch1/ip/jacctor/LiTS/LiTS/TrainingBatch1/volume-1.nii')
        imageheader = imagepredict.header
        numpypredict = imagepredict.get_data().astype(IMG_DTYPE)
        assert numpypredict.shape[0:2] == (512,512)
        nslice = numpypredict.shape[2]
        print('nslice ',nslice)
        image = skimage.transform.resize(numpypredict,
            (options.trainingresample, options.trainingresample, nslice),
            order=0,
            preserve_range=True,
            mode='constant').astype(IMG_DTYPE).transpose(2,1,0)

        slyce = image[61,:,:]
        slycenii = nib.Nifti1Image(slyce, None, header=imageheader)
        slycenii.to_filename(loc+'img_in.nii.gz')

        return slyce, imageheader



def load_outputs(fdict=None, loc=options.outdir, img=None, ih=None):

        if type(fdict) == type(None):
           raise Exception("no dicts passed")
        if type(img) == type(None):
           raise Exception("no img input passed")

        os.system('mkdir -p ' + loc+'img_out/')

        datalist = []
        datalabels = []
        outputarraylist = []
        loclist = []

        image = np.zeros((1,img.shape[0],img.shape[1],1))
        image[0,:,:,0] = img

        for lname in fdict:

                if lname[0:5]=='input':
                    continue
                elif lname=='activation_1' or lname=='dense_1':
                    f = fdict[lname]
                    output = f([image, 0.])
                    print(lname, '\t', output[0].shape)
                    
                    for o in range(output[0].shape[-1]):
                        img_o_np = output[0][0,:,:,o]
                        img_o_nii = nib.Nifti1Image(img_o_np[...,np.newaxis], None, header=ih)
                        img_o_nii.to_filename( loc+'img_out/'+lname+'-'+str(o)+'.nii.gz')

                else:

                    outloc = loc +  lname  +'-'
                    loclist.append(outloc)

                    outputlist = []

                    f = fdict[lname]
                    output = f([image, 0.])
                    print(lname, '\t', output[0].shape)
          
                    for o in range(output[0].shape[-1]):
                        datalabels.append((lname, o))
                        datalist.append(output[0][0,:,:,o].flatten())
                        outputlist.append(output[0][0,:,:,o].flatten())

                        # save each image output
                        img_o_np = output[0][0,:,:,o]
                        img_o_nii = nib.Nifti1Image(img_o_np[...,np.newaxis], None, header=ih)
                        img_o_nii.to_filename( loc+'img_out/'+lname+'-'+str(o)+'.nii.gz')

                    outputlist = np.array(outputlist)
                    outputarraylist.append(outputlist)

        datamatrix = np.array(datalist)
        print(datamatrix.shape)
        return datamatrix, outputarraylist, loclist



def cluster_and_plot(datamatrix, loc=options.outdir, nclusters=options.clusters):
        kernelnorms = np.linalg.norm(datamatrix, axis=1)
        plt.hist(kernelnorms)
        plt.savefig(loc+"hist-"+str(nclusters)+".png", bbox_inches="tight")
        if options.show_hist:
            plt.show()
        else:
            plt.clf()
            plt.close()


        kmeans  = KMeans(n_clusters=nclusters, random_state=0)
        pred    = kmeans.fit_predict(datamatrix)
        clscore = cluster_score(datamatrix, pred)
        print("Size:\t", datamatrix.shape, end=' ')
        print("\tFit:\t", kmeans.score(datamatrix) / datamatrix.shape[0], end=' ')
        print("\tCluster score:\t", clscore)


        if options.tsne:
            tsne = TSNE(init='pca')
            tsne_results = tsne.fit_transform(datamatrix)
            x0 = tsne_results[:,0]
            x1 = tsne_results[:,1]

            plt.scatter(x0,x1,s=9,c=pred)

            plt.savefig(loc+"cluster-TSNE-"+str(nclusters)+".png", bbox_inches="tight")
            if options.show_clusters:
                plt.show()
            else:
                plt.clf()
                plt.close()

        else:

            datamean = np.mean(datamatrix, axis=0)
            datamatrix -= datamean
            
            U, S, Vt = np.linalg.svd(datamatrix)
            proj = np.dot(datamatrix, Vt[0:2, :].T)   
            correction = np.dot(datamean, Vt[0:2, :].T)
            x0 = proj[:,0] + correction[0]
            x1 = proj[:,1] + correction[1]
            plt.scatter(x0,x1, s=9, c=pred)

            plt.savefig(loc+"cluster-"+str(nclusters)+".png", bbox_inches="tight")
            if options.show_clusters:
                plt.show()
            else:
                plt.clf()
                plt.close()

        return clscore


def check_scores_many_clusters(fdict=None, loc=options.outdir, nclustlist=[options.clusters], img=None, ih=None):
    score_array = []
    for nc in nclustlist:

        print("\n Clustering with", nc, "clusters...")

        data, klist, loclist = load_outputs(fdict=fdict, loc=loc, img=img, ih=ih)
        clscores = []
        nlayers = len(klist)
        for kidx, k in enumerate(klist):
            cs = cluster_and_plot(k, loc=loclist[kidx], nclusters=nc) 
            clscores.append(cs)
        score_array.append(clscores)


    for lidx in range(nlayers):
        sc = []
        for cidx, nc in enumerate(nclustlist):
            sc.append(score_array[cidx][lidx])
        plt.plot(nclustlist, sc)
        plt.title("Scores for layer "+str(lidx))
        plt.xlabel("number of clusters")
        plt.ylabel("score")
        plt.savefig(loc+"scores-layer"+str(lidx)+".png", bbox_inches="tight")
        plt.show()


loc = options.outdir
loc += '/layeroutputs/' 
os.system('mkdir -p ' + loc)

imgslice, imageheader = create_slice(loc)
mdict, odict, fdict = load_model(livermodel=options.predictlivermodel)

if not options.test_range_clusters:
    data, klist, loclist = load_outputs(fdict = fdict, loc=loc, img=imgslice, ih=imageheader)
    for kidx, k in enumerate(klist):
        cluster_and_plot(k, loc=loclist[kidx], nclusters=options.clusters)
    cluster_and_plot(data, loc=loc, nclusters=options.clusters)
else:
    nlist = list(range(2,16))
    check_scores_many_clusters(fdict=fdict, loc=loc, nclustlist=nlist, img=imgslice, ih=imageheader)

