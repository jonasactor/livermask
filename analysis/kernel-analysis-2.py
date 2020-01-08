import numpy as np
import os
import sys
import keras
from keras.models import model_from_json, load_model
from optparse import OptionParser # TODO update to ArgParser (python2 --> python3)
import matplotlib as mptlib
mptlib.use('TkAgg')
import matplotlib.pyplot as plt



###
### CAN ONLY BE RUN ON MAEDA (AS OF 11/21/19)
###


### if model is a liver model, from liverhcc code
### otherwise this needs to be changed to tumorhcc
sys.path.append('liverhcc')
from mymetrics import dsc_l2, dsc, dsc_int, l1

sys.setrecursionlimit(5000)

# setup command line parser to control execution
parser = OptionParser()
parser.add_option( "--model",
                  action="store", dest="model", default=None,
                  help="model location", metavar="PATH")
parser.add_option( "--outdir",
                  action="store", dest="outdir", default="./",
                  help="out location", metavar="PATH")
parser.add_option( "--show",
                  action="store_true", dest="show", default=False,
                  help="show plotted clusters", metavar="bool")
parser.add_option( "--use_bias",
                  action="store_true", dest="use_bias", default=False,
                  help="incorporate bias into linear operator", metavar="bool")
(options, args) = parser.parse_args()


# raw dicom data is usually short int (2bytes) datatype
# labels are usually uchar (1byte)
IMG_DTYPE = np.int16
SEG_DTYPE = np.uint8



_globalexpectedpixel=512
_nx = 256 
_ny = 256



def load_modeldict(livermodel=options.model):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    loaded_liver_model=load_model(livermodel, custom_objects={'dsc_l2':dsc_l2, 'l1':l1, 'dsc':dsc, 'dsc_int':dsc})

    layer_dict = dict([(layer.name, layer) for layer in loaded_liver_model.layers])
    model_dict = dict([(layer.name, layer) for layer in layer_dict['model_1'].layers])

    print(loaded_liver_model.summary())
    print(layer_dict['model_1'].summary())

    return model_dict


def build_graph(model_dict):
    for lname in model_dict:
        layer = model_dict[lname]
        print(lname, end='\t')
        print('\t', 'layers in:\t', [n.get_config()['inbound_layers'] for n in layer._inbound_nodes])
#        print('\t', [n.get_config()['outbound_layer'] for n in layer._inbound_nodes])
#        print('\t', [n.get_config()['inbound_layers'] for n in layer._outbound_nodes])
#        print('\t', [n.get_config()['outbound_layer'] for n in layer._outbound_nodes])


class LayerNode():
    def __init__(self, layer, lname):
        self.name = lname
        self.layer = layer
        self.layers_in = [n.get_config()['inbound_layers'] for n in layer._inbound_nodes][0]
        self.myconst = 1.0
        self.prevconst = 1.0
        self.netconst = 1.0
        self.prev_computed = False

        if isinstance(layer, keras.layers.DepthwiseConv2D):
            k = layer.get_weights()[0]
            self.myconst = np.array([np.sum(np.abs(k[:,:,j,0])) for j in range(k.shape[2])])
#            self.myconst = np.array([1.0 for j in range(k.shape[2])])

        if isinstance(layer, keras.layers.AveragePooling2D):
            self.myconst = 0.5

        if isinstance(layer, keras.layers.Conv2D) and not isinstance(layer, keras.layers.DepthwiseConv2D):
            k = layer.get_weights()[0]
#            self.myconst = np.array([np.sum(np.abs(k[:,:,:,j])) for j in range(k.shape[3])])
            self.myconst = np.array([1.0 for j in range(k.shape[3])])

        if isinstance(layer, keras.layers.Dense):
            self.myconst = np.linalg.norm(layer.get_weights()[0][:,0])

        if isinstance(self.layer, keras.layers.InputLayer):
            self.prev_computed = True

    def __repr__(self):
        return self.name
    def __str__(self):
        return self.name


    def get_lip_const(self, nodedict):
        if not self.prev_computed:  
            self.prev_computed = True
            if len(self.layers_in) == 1:
                self.prevconst = [nodedict[l].get_lip_const(nodedict) for l in self.layers_in][0]
            elif len(self.layers_in) == 2:
                self.prevconst = np.add( *[nodedict[l].get_lip_const(nodedict) for l in self.layers_in])
            else:
                self.prevconst = np.inf
            self.netconst = np.multiply(self.myconst, self.prevconst)

        return self.netconst

def get_node_dict(mdict):
    ndict = dict()
    for lname in mdict:
        ndict[lname] = LayerNode(mdict[lname], lname)

    return ndict


def load_kernels(mdict=None, loc=options.outdir, use_bias=options.use_bias):

        if type(mdict) == type(None):
           raise Exception("no dicts passed")
        
        kernellist = []
        kernelbiasedlist = []
        kernellabels = []
        kernelinshapes = []
        loclist = []

        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

        for l2 in mdict:
            print(l2)
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
                print('\n')
 
            elif l2[0:16] == 'depthwise_conv2d':
                outloc  = loc + '/' + l2  +'-'
                kernel  = mdict[l2].get_weights()[0]
                bias    = mdict[l2].get_weights()[1]
                inshape = mdict[l2].input_shape
    
                assert inshape[-1] == kernel.shape[2]

                for j in range(kernel.shape[2]):
                    ker = kernel[:,:,j,0]
                    ker = ker[...,np.newaxis,np.newaxis]
                    bis = bias[j]
                    ins = (inshape[0], inshape[1], inshape[2], 1)
                    otl = outloc + str(j) + '-'

                    kernellist.append(ker) 
                    kernellabels.append(l2)
                    kernelinshapes.append(ins[1:-1])
                    loclist.append(otl)

                print("Layer", l2, "has a kernel of size", kernel.shape, "with input of shape", inshape)
                print('\n')
        return kernellist, kernellabels, kernelinshapes, loclist, kernelbiasedlist



# as according to Segdhi et al
# input_shape : feature map to be convolved
def singular_values(kernel, input_shape):
    transforms = np.fft.fft2(kernel, input_shape, axes=[0,1])
    return np.linalg.svd(transforms)

# as according to Segdhi et al
def plot_singular_values(kernel, inshape, loc=options.outdir, b=False):
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

mdict = load_modeldict(livermodel=options.model)
build_graph(mdict)
nd = get_node_dict(mdict)
nd['dense_1'].get_lip_const(nd)
for nname in nd:
    print(nname)
    print('\t', nd[nname].myconst)
    print('\t', nd[nname].netconst)
    print('\n\n')

#klist, klabels, kinshapes, loclist, kblist = load_kernels(mdict = mdict, loc=loc)
#for kidx, k in enumerate(klist):
#    plot_singular_values(k, kinshapes[kidx], loc=loclist[kidx], b=False)
##    cluster_and_plot(k, loc=loclist[kidx])

