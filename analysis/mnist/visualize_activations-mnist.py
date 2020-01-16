import numpy as np
import csv
import os

import keras
from keras.models import load_model, Model
import keras.backend as K

import matplotlib.pyplot as plt

import sys
sys.path.append("~/Bioinformatics/image-segmentation/Code/test-Keras")
from ista import ISTA

###
#$$ nonconvex loss function
###
def not_convex(y_true, y_pred):
    return K.sum(K.square(y_true - y_pred)) / ( K.sum(K.square(y_true)) + K.sum(K.square(y_pred)) )



def make_viz_model(modelloc):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    loaded = load_model(modelloc, custom_objects={'ISTA':ISTA, 'not_convex':not_convex})
    model_dict = dict([(layer.name, layer) for layer in loaded.layers])

    m = loaded
    m_names = [layer.name for layer in m.layers]

    viz_outputs = Model( inputs  = m.layers[0].input,  outputs = [layer.output for layer in m.layers[1:]])

    return viz_outputs, m_names, model_dict


def viz_activations(vizmodel, m_names, mdict, loc):
    print(loc)
    for layer_name in m_names:
        lyr = mdict[layer_name]
        if isinstance(lyr, keras.layers.Dense):
            k = lyr.get_weights()[0]
            print(k.shape, np.sum( np.abs(k) > 0), np.sum(np.abs(k) > 0.0001), np.sum(np.abs(k) > 0.001))
            plt.imshow(k, cmap='gray')
            plt.savefig(loc+"kernel-"+layer_name+".png", bbox_inches="tight")
            plt.clf()
            plt.close()


outloclist   = [ '/home/jonasactor/Bioinformatics/image-segmentation/Code/test-Keras/viz-l1/' ,
                 '/home/jonasactor/Bioinformatics/image-segmentation/Code/test-Keras/viz-ista/' ]
modelloclist = [ '/home/jonasactor/Bioinformatics/image-segmentation/Code/test-Keras/mnistmodel-l1.h5',
                 '/home/jonasactor/Bioinformatics/image-segmentation/Code/test-Keras/mnistmodel-ista.h5' ]


for j in range(len(outloclist)):
    modelloc = modelloclist[j]
    outloc   = outloclist[j]
    vzm, names, mdict = make_viz_model(modelloc)
    viz_activations(vzm, names, mdict, outloc)
