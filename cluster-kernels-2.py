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
#parser.add_option( "--rootlocation",
#                  action="store", dest="rootlocation", default='/rsrch1/ip/jacctor/LiTS/LiTS',
#                  help="root location for images for training", metavar="Path")
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


def get_known_kernels():
    klist  = []
    klabel = []
    
    # Mean
    klist.append((1./9)*np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]))
    klabel.append("mean")

    # Laplacian 1
    klist.append(0.125*np.array([0, 1, 0, 1, -4, 1, 0, 1, 0]))
    klabel.append("laplacian1")

#    # Laplacian 2
#    klist.append(0.0625*np.array([1, 1, 1, 1, -8, 1, 1, 1, 1]))
#    klabel.append("laplacian2")
    
#    # Laplacian Diag
#    klist.append(0.125*np.array([1, 0, 1, 0, -4, 0, 1, 0, 1]))
#    klabel.append("laplacian-diag")

    # Gaussian Blur
    klist.append(0.0625*np.array([1, 2, 1, 2, 4, 2, 1, 2, 1]))
    klabel.append("gaussian")

    # Edge Right Sobel
    klist.append(0.125*np.array([-1, 0, 1, -2, 0, 2, -1, 0, 1]))
    klabel.append("edge-right")

#    # Edge Up
#    klist.append(0.125*np.array([1, 2, 1, 0, 0, 0, -1, -2, -1]))
#    klabel.append("edge-up")

#    # Edge Left
#    klist.append(0.125*np.array([1, 0, -1, 2, 0, -2, 1, 0, -1]))
#    klabel.append("edge-left")

#    # Edge Down
#    klist.append(0.125*np.array([-1, -2, -1, 0, 0, 0, 1, 2, 1]))
#    klabel.append("edge-down")

    # Identity
    klist.append(np.array([0, 0, 0, 0, 1, 0, 0, 0, 0]))
    klabel.append("identity")

    # Sharpen
    klist.append((1./9)*np.array([0, -1, 0, -1, 5, -1, 0, -1, 0]))
    klabel.append("sharpen")


    print(klabel)
    return klist, klabel



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
        try:
            model_dict = dict([(layer.name, layer) for layer in loaded_liver_model.layers])
            return model_dict
        except:
            raise Exception("1-gpu version not implemented yet")


def load_kernels_as_operators(mdict=None, loc=options.outdir):

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
                print("Layer", l2, "has kernels of size", kernel.shape)

        return kernellist, kernellabels, kernelinshapes, loclist

def load_kernels_as_datamatrix(mdict=None, loc=options.outdir):

        if type(mdict) == type(None):
           raise Exception("no dicts passed")
        
        datalist = []
        datalabels = []
        kernelarraylist = []
        loclist = []

        for l2 in mdict:
            if l2[0:6] == 'conv2d':
     
                outloc = loc +  l2  +'-'
                kernellist = []
                kernel = mdict[l2].get_weights()[0]
                bias   = mdict[l2].get_weights()[1]

                for o in range(kernel.shape[-1]):
                    for i in range(kernel.shape[-2]):

                        datalabels.append((l2[7:],i,o))
                        
                        if options.normalize:
                            datalist.append(kernel[:,:,i,o].flatten()*np.sign(kernel[1,1,i,o]))
                            kernellist.append(kernel[:,:,i,o].flatten()*np.sign(kernel[1,1,i,o]))
                        else:
                            datalist.append(kernel[:,:,i,o].flatten())
                            kernellist.append(kernel[:,:,i,o].flatten())

                if options.normalize:
                     kernellist = np.array(kernellist)
                     rownorm = np.linalg.norm(kernellist, axis=1)
                     rowzero = rownorm <  1.e-6
                     rownzro = rownorm >= 1.e-6
                     if np.sum(rowzero) > 0:
                         print("\tThere are",np.sum(rowzero),"zeroes.")
                         kernellist[rowzero,:] = np.zeros((rowzero.shape[0],9))
                     kernellist[rownzro] /= rownorm[:, np.newaxis]

                kernellist = np.array(kernellist)
                kernelarraylist.append(kernellist)
                loclist.append(outloc)
                print("Layer", l2, "has kernels for a matrix size", kernellist.shape)
        datamatrix = np.array(datalist)
        print(datamatrix.shape)
        return datamatrix, kernelarraylist, loclist

def singular_values(kernel, input_shape):
    transforms = np.fft.fft2(kernel, input_shape, axes=[0,1])
    return np.linalg.svd(transforms)

def cluster(datamatrix, nclusters=options.clusters):
    kmeans  = KMeans(n_clusters=nclusters, random_state=0)
    pred    = kmeans.fit_predict(datamatrix)
    clscore = cluster_score(datamatrix, pred)
    print("\tFit:\t", kmeans.score(datamatrix) / datamatrix.shape[0], end=' ')
    print("\tCluster score:\t", clscore)
    return pred

def plot_singular_values(D, loc=options.outdir):
    n_el = np.prod(D.shape)
#    Dflat = D.flatten()
#    Dflat[::-1].sort()
#    n_el = D.shape[0]
    plt.plot(list(range(n_el)), D)
    plt.savefig(loc+".png", bbox_inches="tight")
    if options.show:
        plt.show()
    else:
        plt.clf()
        plt.close()


def perform_kernel_comparison_3(kernel, inshape, loc, knownlist, knownlabels):
    num_inchannels  = kernel.shape[2]
    num_outchannels = kernel.shape[3]

    knownsvds = []
    knowncounts = []
    for kidx, k in enumerate(knownlist):
        ksquare = k.reshape(3,3)
        U2, D2, Vt2 = singular_values(ksquare[...,np.newaxis,np.newaxis], inshape)
        D2 = D2.flatten()
        D2[::-1].sort()
        D2 /= D2[0]
        knownsvds.append(D2)
        knowncounts.append(0)
    knowncounts.append(0)

    for o in range(num_outchannels):
        for i in range(num_inchannels):

            kio = kernel[:,:,i,o]
            U1, D1, Vt1 = singular_values(kio[...,np.newaxis,np.newaxis], inshape)
            D1 = D1.flatten()
            D1[::-1].sort()
            D1 /= D1[0]

            minval = [np.inf, '']
            for kidx, k in enumerate(knownlist):
                DD = knownsvds[kidx]
                compval = np.linalg.norm(D1-DD, ord=1)
                if compval < minval[0]:
                    minval[0] = compval
                    minval[1] = knownlabels[kidx]

            if minval[0] > 0.10 * np.prod(inshape):
                minval[1] = 'indeterminate'

            if minval[1] == 'indeterminate':
               knowncounts[-1] += 1
            else:
               for kidx, k in enumerate(knownlist):
                   if minval[1] == knownlabels[kidx]:
                       knowncounts[kidx] += 1
                       break

    print('\t', sum(knowncounts), '\t', [ x /sum(knowncounts) for x in  knowncounts])           


def perform_channel_comparison_2(kernel, inshape, loc, knownlist, knownlabels):
   num_outchannels = kernel.shape[3]
   for o in range(num_outchannels):

        ko = kernel[...,o]
        U1, D1, Vt1 = singular_values(ko[...,np.newaxis], inshape)
        D1 = D1.flatten()
        D1[::-1].sort()
        D1 /= D1[0]
        plot_singular_values(D1, loc=loc+'out-'+str(o)+'-spectrum')

        minval = [np.inf, '']
        maxval = [-np.inf, '']
        for kidx, k in enumerate(knownlist):
            ksquare = k.reshape(3,3)
            num_outchannels = kernel.shape[3]
            kcopied = np.zeros(kernel.shape[:3])
            for i in range(kernel.shape[2]):
                kcopied[:,:,i] = ksquare

            U2, D2, Vt2 = singular_values(kcopied[...,np.newaxis], inshape)
            D2 = D2.flatten()
            D2[::-1].sort()
            D2 /= D2[0]
            plot_singular_values(D1 - D2, loc=loc+'out-'+str(o)+'-'+knownlabels[kidx]+'-normdiff')
           
            compval = np.linalg.norm(D1-D2, ord=1)
            if compval < minval[0]:
                minval[0] = compval
                minval[1] = knownlabels[kidx]
            if compval > maxval[0]:
                maxval[0] = compval
                maxval[1] = knownlabels[kidx]
        print('\t outchannel '+str(o)+' is closest to '+minval[1]+" with score "+str(minval[0])+' and is farthest from '+maxval[1]+' with score '+str(maxval[0]))


def perform_svd_comparison(kernel, inshape, loc, knownlist, knownlabels):

    for kidx, k in enumerate(knownlist):
        ksquare = k.reshape(3,3)
        kcopied = np.zeros_like(kernel)
        for i in range(kernel.shape[2]):
            for j in range(kernel.shape[3]):
                kcopied[:,:,i,j] = ksquare
        diff = kernel - kcopied

        U1, D1, Vt1 = singular_values(kernel, inshape)
        U2, D2, Vt2 = singular_values(kcopied, inshape)


        D1 = D1.flatten()
        D1[::-1].sort()
        D1 /= D1[0]
 
        D2 = D2.flatten()
        D2[::-1].sort()
        D2 /= D2[0]

        plot_singular_values(D1,      loc=loc)
        plot_singular_values(D1 - D2, loc=loc+knownlabels[kidx]+'-difference-of-nucnorms-')
        print("\t kernel - "+knownlabels[kidx]+" has nuclear norm ",np.linalg.norm(D1 - D2, ord=1))

def get_proj(datamatrix):

    datamean = np.mean(datamatrix, axis=0)
    datamatrix -= datamean

    U, S, Vt   = np.linalg.svd(datamatrix)
    
    datamatrix += datamean
    return U, Vt, datamean

def get_proj_clinical(kklist):
    if len(kklist) == 2:
        Vt = np.vstack(kklist)
        print(Vt.shape)
        return Vt    
    else:
        U, Vt, datamean = get_proj( np.vstack(kklist) )
        return Vt


def plot_clusters(datamatrix, pred, Vt, datamean, loc=options.outdir, nclusters=options.clusters):

    proj       = np.dot(datamatrix, Vt[0:2, :].T)   
    correction = np.dot(datamean, Vt[0:2, :].T)

    x0 = proj[:,0] + correction[0]
    x1 = proj[:,1] + correction[1]
    plt.scatter(x0,x1, s=9, c=pred)

    known, klabel = get_known_kernels()
    known = np.array(known)
           
    known -= datamean

    known = np.dot(known, Vt[0:2, :].T)
    k0 = known[:,0] + correction[0]
    k1 = known[:,1] + correction[1]
    plt.scatter(k0, k1, s=36, c='red', marker="s")
    for i, txt in enumerate(klabel):
        plt.annotate(txt, (k0[i], k1[i])) 

    plt.savefig(loc+"cluster-"+str(nclusters)+".png", bbox_inches="tight")
    if options.show:
        plt.show()
    else:
        plt.clf()
        plt.close()





loc = options.outdir
if options.normalize:
    loc += '/normalized/'
else:
    loc += '/original/' 
os.system('mkdir -p ' + loc)

###
### load model
###
mdict = load_model(livermodel=options.predictlivermodel)

###
### kernel clustering and projection
###

data, klist, loclist = load_kernels_as_datamatrix(mdict = mdict, loc=loc)
U, Vt, datamean = get_proj(data)

knownlist, knownlabels = get_known_kernels()
#Ct = get_proj_clinical(knownlist)
#zeromean = np.zeros_like(datamean)

for kidx, k in enumerate(klist):
    pred = cluster(k, nclusters=options.clusters)
    plot_clusters(k, pred, Vt, datamean, loc=loclist[kidx],             nclusters=options.clusters)
#    plot_clusters(k, pred, Ct, zeromean, loc=loclist[kidx]+'clinical-', nclusters=options.clusters) 

### kernel operator analysis
klist, klabels, kinshapes, loclist = load_kernels_as_operators(mdict=mdict, loc=loc)

for kidx, k in enumerate(klist):
    print('conv2d_'+str(kidx), end=' ')
    perform_kernel_comparison_3(k, kinshapes[kidx], loclist[kidx], knownlist, knownlabels)

# channel level
#for kidx, k in enumerate(klist):
#    print('conv2d_'+str(kidx))
#    perform_channel_comparison_2(k, kinshapes[kidx], loclist[kidx], knownlist, knownlabels)

# layer level
#for kidx, k in enumerate(klist):
#    print('conv2d_'+str(kidx))
#    perform_svd_comparison(k, kinshapes[kidx], loclist[kidx], knownlist, knownlabels)

for kidx, k in enumerate(knownlist):
    kten = k.reshape((3,3))
    U, D, Vt = singular_values(kten[...,np.newaxis,np.newaxis], (256,256))
    Dflat = D.flatten()
    Dflat[::-1].sort()
    n_el = np.prod(D.shape)
    plt.plot(list(range(n_el)), Dflat)
    plt.savefig(loc+knownlabels[kidx]+".png", bbox_inches="tight")
    if options.show:
        plt.show()
    else:
        plt.clf()
        plt.close()

