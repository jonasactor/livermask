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
parser.add_option( "--activation",
                  action="store", dest="activation", default='relu',
                  help="activation function", metavar="string")
parser.add_option( "--segthreshold",
                  type="float", dest="segthreshold", default=0.5,
                  help="cutoff for binary segmentation from real-valued output", metavar="float")
parser.add_option( "--predictlivermodel",
                  action="store", dest="predictlivermodel", default="/rsrch1/ip/jacctor/livermask/unet-2gpu-alldata-depth3/001/000/liver/modelunet.h5",
                  help="model weights (.h5) for liver seg prediction", metavar="Path")
parser.add_option( "--show",
                  action="store_true", dest="show", default=False,
                  help="show plotted clusters", metavar="bool")
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
        layer_dict = dict([(layer.name, layer) for layer in loaded_liver_model.layers])
        model_dict = dict([(layer.name, layer) for layer in layer_dict['model_1'].layers])
        return model_dict
    else:
        raise Exception("1-gpu version not implemented yet")



def load_kernels(mdict=None, loc=options.outdir):

        if type(mdict) == type(None):
           raise Exception("no dicts passed")
        
        kernellist = []
        kernellabels = []
        kernelinshapes = []
        loclist = []

        for l2 in mdict:
            if l2[0:6] == 'conv2d':

                outloc  = loc + '/' + l2  +'-'
                kernel  = mdict[l2].get_weights()[0]
                bias    = mdict[l2].get_weights()[1]
                inshape = mdict[l2].input_shape
    
                assert inshape[-1] == kernel.shape[2]

                kernellist.append(kernel) 
                kernellabels.append(l2)
                kernelinshapes.append(inshape[1:-1])
                loclist.append(outloc)
                print("Layer", l2, "has a kernel of size", kernel.shape, "with input of shape", inshape)

        return kernellist, kernellabels, kernelinshapes, loclist

# input_shape : feature map to be convolved
def singular_values(kernel, input_shape):
    transforms = np.fft.fft2(kernel, input_shape, axes=[0,1])
    return np.linalg.svd(transforms)


def plot_singular_values(kernel, inshape, loc=options.outdir):
    U, D, Vt = singular_values(kernel, inshape)
    n_el = np.prod(D.shape)
    Dflat = D.flatten()
    Dflat[::-1].sort()
    plt.plot(list(range(n_el)), Dflat)
    plt.savefig(loc+"spectrum.png", bbox_inches="tight")
    if options.show:
        plt.show()
    else:
        plt.clf()
        plt.close()


def get_known_kernels():
    klist  = []
    klabel = []
    
    # Mean
    klist.append((1./9)*np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]))
    klabel.append("mean")

    # Laplacian 1
    klist.append(0.125*np.array([0, 1, 0, 1, -4, 1, 0, 1, 0]))
    klabel.append("laplacian1")

    # Laplacian 2
    klist.append(0.0625*np.array([1, 1, 1, 1, -8, 1, 1, 1, 1]))
    klabel.append("laplacian2")
    
    # Laplacian Diag
    klist.append(0.125*np.array([1, 0, 1, 0, -4, 0, 1, 0, 1]))
    klabel.append("laplacian-diag")

    # Gaussian Blur
    klist.append(0.0625*np.array([1, 2, 1, 2, 4, 2, 1, 2, 1]))
    klabel.append("gaussian")

    # Edge Right
    klist.append(0.125*np.array([-1, 0, 1, -2, 0, 2, -1, 0, 1]))
    klabel.append("edge-right")

    # Edge Up
    klist.append(0.125*np.array([1, 2, 1, 0, 0, 0, -1, -2, -1]))
    klabel.append("edge-up")

    # Edge Left
    klist.append(0.125*np.array([1, 0, -1, 2, 0, -2, 1, 0, -1]))
    klabel.append("edge-left")

    # Edge Down
    klist.append(0.125*np.array([-1, -2, -1, 0, 0, 0, 1, 2, 1]))
    klabel.append("edge-down")

    # Identity
    klist.append(np.array([0, 0, 0, 0, 1, 0, 0, 0, 0]))
    klabel.append("identity")

    # Sharpen
    klist.append((1./9)*np.array([0, -1, 0, -1, 5, -1, 0, -1, 0]))
    klabel.append("sharpen")

    return klist, klabel


loc = options.outdir
os.system('mkdir -p ' + loc)

mdict = load_model(livermodel=options.predictlivermodel)
klist, klabels, kinshapes, loclist = load_kernels(mdict = mdict, loc=loc)
for kidx, k in enumerate(klist):
    plot_singular_values(k, kinshapes[kidx], loc=loclist[kidx])
#    cluster_and_plot(k, loc=loclist[kidx])

