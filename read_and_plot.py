import numpy as np
import csv
import sys
import os
import json
import nibabel as nib
from optparse import OptionParser
from scipy import ndimage
from PIL import Image
import matplotlib as mptlib
import glob
mptlib.use('TKAgg')
import matplotlib.pyplot as plt

sys.setrecursionlimit(5000)

parser = OptionParser()
parser.add_option( "--imgloc",
                   action="store", dest="imgloc", default='./',
                   help="location of images to plot in square", metavar="Path")
(options, args) = parser.parse_args()

IMG_DTYPE = np.int16
SEG_DTYPE = np.uint8


types_of_layers = map( lambda x: x.split('/')[-1].split('-')[-2][:-2], glob.glob(options.imgloc+"*_1-0.nii.gz"))
for layertype in types_of_layers:
    if layertype == "dense" or layertype == "spatial_dropout2d" or layertype == "activation":
        continue
    eachlayer = glob.glob(options.imgloc+layertype+"*-0.nii.gz")
    print(layertype, len(eachlayer))
    for lnum in range(1, len(eachlayer)+1):
        
        loc_list = sorted(glob.glob(options.imgloc+layertype+'_'+str(lnum)+'-*.nii.gz'), key = lambda x: float(x.split('-')[-1].split('.')[0]))
        nii_map  = map(nib.load, loc_list)
        img_map  = map(lambda x: x.get_data().astype(IMG_DTYPE), nii_map)
        img_list = list(img_map)

        if len(img_list) <= 16:
            ncols = 4
        else:
            ncols = 8
        nrows=  int(len(img_list) / ncols)

        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*1.25,nrows*1.25), subplot_kw={'xticks': [], 'yticks': []})

        for ax, imm, imname in zip(axs.flat, img_list, loc_list):
            ax.imshow(imm[...,0], cmap=plt.cm.bone)
            ax.set_title(imname.split('-')[-1].split('.')[0])
        plt.suptitle(layertype + ' '+str(lnum))
        plt.tight_layout()
        plt.savefig(options.imgloc+layertype+'-'+str(lnum)+"_composite.png", bbox_inches="tight")
#        plt.show()
        plt.close()




