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
sys.setrecursionlimit(5000)
sys.path.append("/rsrch1/ip/jacctor/livermask/liverhcc")
from mymetrics import dsc, dsc_l2, l1
from ista import ISTA
import preprocess

from aux import *


def viz_at_each_layer(modelloc, img_in, m_names, mdict, outloc):
    vzm, names, mdict = make_viz_model(modelloc)
    predict_viz_model(vzm, img_in, m_names, mdict, outloc)



# make a model that outputs the output of every layer
def make_viz_model(modelloc):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    loaded = load_model(modelloc, custom_objects={'dsc_l2':dsc_l2, 'l1':l1, 'dsc':dsc, 'dsc_int':dsc})
    layer_dict = dict([(layer.name, layer) for layer in loaded.layers])
    model_dict = dict([(layer.name, layer) for layer in layer_dict['model_1'].layers])
    
    m = layer_dict['model_1']
    m_names = [layer.name for layer in layer_dict['model_1'].layers]

    viz_outputs = Model( inputs  = m.layers[0].input,  outputs = [layer.output for layer in m.layers[1:]])

    return viz_outputs, m_names[1:], model_dict

# predict and visualize the output at every layer
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



#imgloc   = '/rsrch1/ip/jacctor/LiTS/LiTS/TrainingBatch2/volume-110.nii'
#img = get_img(imgloc)
#
#outloclist   = [ '/rsrch1/ip/jacctor/livermask/analysis/test_augment/' ]
#modelloclist = [ '/rsrch1/ip/jacctor/livermask/liverhcc/test_augment/005/003/liver/modelunet.h5' ]
#
#
#for j in range(len(outloclist)):
#    modelloc = modelloclist[j]
#    outloc   = outloclist[j]
#    vzm, names, mdict = make_viz_model(modelloc)
#    predict_viz_model(vzm, img, names, mdict, outloc)



# histogram of image pixel values
def plot_histogram(data, b=100, r=(-110,410)):
    counts, bin_edges = np.histogram(data, bins=b, range=r)
    plt.bar(bin_edges[:-1], counts, width=[0.8*(bin_edges[i+1]-bin_edges[i]) for i in range(len(bin_edges)-1)])
    plt.show()



# scatterplot of accuracy vs size
def plot_dsc_vs_size(modelloc, imgloc):
    im, sg, pred, pred_seg = makeprediction(modelloc, imgloc)
    _,_,_, dsc_int_2D = compute_dsc_scores(sg, pred_seg)

    dims = sg.shape
    areas = [None]*dims[0]
    for iii in range(dims[0]):
        areas[iii] = np.sum(np.abs(sg[iii,...]))

    plt.figure()
    plt.scatter(areas, dsc_int_2D, marker='o')
    plt.show()





def load_kernels(livermodel, loc):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    loaded_liver_model = load_model(livermodel, custom_objects={'dsc_l2':dsc_l2, 'l1':l1, 'dsc':dsc, 'dsc_int':dsc, 'ISTA':ISTA})


    if options.gpu > 1:
        layer_dict = dict([(layer.name, layer) for layer in loaded_liver_model.layers])
        model_dict = dict([(layer.name, layer) for layer in layer_dict['model_1'].layers])

        datalist = []
        datalabels = []
        kernellistlist = []
        for l2 in model_dict:
            if l2[0:6] == 'conv2d':
     
                outloc = loc +  l2  
                kernellist = []
                kernel = model_dict[l2].get_weights()[0]
                bias   = model_dict[l2].get_weights()[1]

                for o in range(kernel.shape[-1]):
                    for i in range(kernel.shape[-2]):

                        if options.normalize:
                            kernellist.append(kernel[:,:,i,o]*np.sign(kernel[1,1,i,o]).flatten())
                        else:
                            kernellist.append(kernel[:,:,i,o].flatten())
                        datalist.append(kernel[:,:,i,o].flatten())
                        datalabels.append((l2[7:],i,o))

                if options.normalize:
                     kernellist = np.array(kernellist)
                     print(kernellist.shape)
                     rownorm = np.linalg.norm(kernellist, axis=1)
                     rowzero = rownorm <  1.e-6
                     rownzro = rownorm >= 1.e-6
                     print(rownorm.shape)
                     kernellist[rowzero] = np.zeros((rownorm.shape[0],9))
                     kernellist[rownzro] /= rownorm

                kernellistlist.append(kernellist)
                kernellist = np.array(kernellist)
                print("Layer", l2, "has kernels for a matrix size", kernellist.shape)
                cluster_and_plot(kernellist, outloc)
        datamatrix = np.array(datalist)
        print(datamatrix.shape)
        return datamatrix

def cluster_and_plot(datamatrix, loc=options.outdir):
        datamean = np.mean(datamatrix, axis=0)
        datamatrix -= datamean
        
        kmeans = KMeans(n_clusters=options.clusters, random_state=0)
        pred   = kmeans.fit_predict(datamatrix)
        print("Fit:\t", kmeans.score(datamatrix))
        datamatrix += datamean

        U, S, Vt = np.linalg.svd(datamatrix)
        proj = np.dot(datamatrix, Vt[:, 0:2])   
        x0 = proj[:,0]
        x1 = proj[:,1]
        plt.scatter(x0,x1, s=9, c=pred)
        plt.savefig(loc+"cluster-"+str(options.clusters)+".png", bbox_inches="tight")
#        plt.show()

loc = options.outdir
if options.normalize:
    loc += 'normalized/'
else:
    loc += 'original/' 
os.system('mkdir -p ' + loc)
data = load_kernels(livermodel=options.predictlivermodel, loc=loc)
cluster_and_plot(data, loc)
