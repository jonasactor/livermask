import numpy as np
import csv
import os
import nibabel as nib
import skimage.transform
import preprocess

import keras
from keras.models import load_model, Model
import keras.backend as K

import matplotlib as mptlib
mptlib.use('TkAgg')
import matplotlib.pyplot as plt

import sys
sys.path.append("/rsrch1/ip/jacctor/livermask/liverhcc")
from mymetrics import dsc, dsc_l2, l1

def make_viz_model(modelloc):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    loaded = load_model(modelloc, custom_objects={'dsc_l2':dsc_l2, 'l1':l1, 'dsc':dsc, 'dsc_int':dsc})
    layer_dict = dict([(layer.name, layer) for layer in loaded.layers])
    model_dict = dict([(layer.name, layer) for layer in layer_dict['model_1'].layers])
    
    m = layer_dict['model_1']
    m_names = [layer.name for layer in layer_dict['model_1'].layers]

    viz_outputs = Model( inputs  = m.layers[0].input,  outputs = [layer.output for layer in m.layers[1:]])

    return viz_outputs, m_names[1:], model_dict

def predict_viz_model(vizmodel, imgin, m_names, mdict, loc):

    activations = vizmodel.predict(imgin)
    
    print(len(activations))
    print(len(m_names))

    imgs_per_row = 4
    for layer_name, layer_activation in zip(m_names, activations):
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]
        n_cols = n_features // imgs_per_row
        if n_cols < 1:
            n_cols = 1
        display_grid = np.zeros((size * n_cols, imgs_per_row*size))
        for col in range(n_cols):
            for row in range(min(imgs_per_row,n_features)):
                channel_img = layer_activation[0,:,:,col*imgs_per_row + row]
                display_grid[col*size: (col+1)*size, row*size:(row+1)*size] = np.clip(channel_img, -5.0, 20.0)
        scale = 1. / size
        plt.figure(figsize=(scale*display_grid.shape[1], scale*display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='gray')
        plt.savefig(loc+"img-"+layer_name+".png", bbox_inches="tight")
        if n_cols == 1:
            plt.show()
        else:
            plt.clf()
            plt.close()

        lyr = mdict[layer_name]
        print(layer_name)
        if isinstance(lyr, keras.layers.Conv2D) or isinstance(lyr, keras.layers.Dense):
            k = lyr.get_weights()[0]
            if isinstance(lyr, keras.layers.Dense):
                klist = [k[j,0]*np.ones((3,3)) for j in range(k.shape[0])]
            elif isinstance(lyr, keras.layers.DepthwiseConv2D):
                klist = [k[:,:,j,0] for j in range(k.shape[2])]
            else:
                klist = [k[:,:,0,j] for j in range(k.shape[3])]
            n_cols = n_features // imgs_per_row
            if n_cols < 1:
                n_cols = 1
            display_grid = np.zeros((5 * n_cols, imgs_per_row*5))
            for col in range(n_cols):
                for row in range(imgs_per_row):
                    kkk = klist[col*imgs_per_row + row]
                    k_padded = np.zeros((5,5))
                    k_padded[1:4,1:4] = kkk
                    display_grid[col*5: (col+1)*5, row*5:(row+1)*5] = k_padded
            scale = 1. / 5
            plt.figure(figsize=(scale*display_grid.shape[1], scale*display_grid.shape[0]))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='gray', vmin=-2.0, vmax=2.0)
            plt.savefig(loc+"kernel-"+layer_name+".png", bbox_inches="tight")
            plt.clf()
            plt.close()


def get_img(imgloc):

    npimg, _, _ = preprocess.reorient(imgloc)
    npimg       = preprocess.resize_to_nn(npimg, transpose=True).astype(np.int16)
    npimg       = preprocess.window(npimg, -100,300)
    npimg       = preprocess.rescale(npimg, -100,300)

    print(npimg.shape)
    midslice = npimg[int(npimg.shape[0] / 2),:,:]

    return midslice[np.newaxis,:,:,np.newaxis]

imgloc   = '/rsrch1/ip/jacctor/LiTS/LiTS/TrainingBatch2/volume-110.nii'
img = get_img(imgloc)

outloclist   = [ '/rsrch1/ip/jacctor/livermask/analysis/test_augment/',
#                 '/rsrch1/ip/jacctor/livermask/analysis/activations/depthwise-dropout-l1reg/',
                 ]
modelloclist = [ '/rsrch1/ip/jacctor/livermask/liverhcc/test_augment/005/003/liver/modelunet.h5', 
#                 '/rsrch1/ip/jacctor/livermask/liverhcc/dropout-l1reg/005/001/liver/modelunet.h5',
                 ]
for j in range(len(outloclist)):
    modelloc = modelloclist[j]
    outloc   = outloclist[j]
    vzm, names, mdict = make_viz_model(modelloc)
    predict_viz_model(vzm, img, names, mdict, outloc)
