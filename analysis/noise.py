import os
import csv
import time
import json
from optparse import OptionParser

import matplotlib as mptlib
mptlib.use('TkAgg')
import matplotlib.pyplot as plt

import nibabel as nib
import skimage.transform

import keras
import keras.backend as K
import keras.losses
from keras import models
from keras import layers
from keras.models import load_model
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

import tensorflow as tf

import sys
sys.setrecursionlimit(5000)
sys.path.append('/rsrch1/ip/jacctor/livermask/liverhcc')
from mymetrics import dsc_l2, dsc, dsc_int, l1, dsc_int_3D, dsc_l2_3D
from ista import ISTA
import preprocess

import math
import numpy as np
import scipy as sp
import scipy.stats as stats
from scipy.signal import convolve2d

IMG_DTYPE = np.int16
SEG_DTYPE = np.uint8

_globalexpectedpixel=512
_nx = 256
_ny = 256

npx = 256
hu_lb = -100
hu_ub = 400 
std_lb = 0
std_ub = 100



###
### set options
###
parser = OptionParser()
parser.add_option( "--model",
        action="store", dest="model", default=None,
        help="model location", metavar="PATH")
parser.add_option( "--model_reg",
        action="store", dest="model_reg", default=None,
        help="model location", metavar="PATH")
parser.add_option( "--outdir",
        action="store", dest="outdir", default="./",
        help="out location", metavar="PATH")
parser.add_option( "--imgloc",
        action="store", dest="imgloc", default='/rsrch1/ip/jacctor/LiTS/LiTS/TrainingBatch2/volume-124.nii',
        help="img location", metavar="PATH.nii")
parser.add_option( "--segloc",
        action="store", dest="segloc", default='/rsrch1/ip/jacctor/LiTS/LiTS/TrainingBatch2/segmentation-124.nii',
        help="seg location", metavar="PATH.nii")
parser.add_option( "--delta",
        type="float", dest="delta", default=0.1,
        help="perturbation", metavar="float")
parser.add_option( "--test",
        action="store_true", dest="test", default=False,
        help="small sets for testing purposes", metavar="bool")
parser.add_option( "--one_at_a_time",
        action="store_true", dest="one_at_a_time", default=False,
        help="process one image at a time, using a bash script instead of a loop : saves on memory", metavar="bool")
parser.add_option( "--idxOne",
        type="int", dest="idxOne", default=124,
        help="choice for command line image idx", metavar="int")
(options, args) = parser.parse_args()



rootloc_mda = '/rsrch1/ip/dtfuentes/github/RandomForestHCCResponse'
if options.test:
    dbfile_mda = '/rsrch1/ip/jacctor/livermask/trainingdata-mda-small.csv'
else:
    dbfile_mda = '/rsrch1/ip/dtfuentes/github/RandomForestHCCResponse/datalocation/trainingdata.csv'
#    dbfile_mda = '/rsrch1/ip/jacctor/livermask/trainingdata-mda-small.csv'


###
### start a session - will need same session to link K to tf
###

session = tf.Session()
K.set_session(session)

###
### get data
###

def get_img(imgloc, segloc=None):
    npimg, _, npseg = preprocess.reorient(imgloc, segloc)
    npimg           = preprocess.resize_to_nn(npimg).astype(np.float32)
    npimg           = preprocess.window(npimg,  hu_lb, hu_ub)
    npimg           = preprocess.rescale(npimg, hu_lb, hu_ub)
    npseg           = preprocess.resize_to_nn(npseg).astype(np.uint8)

    print(npimg.shape)
    print(npseg.shape)
    assert npimg.shape == npseg.shape
    return npimg[...,np.newaxis], npseg[...,np.newaxis]
###
### WARNING: can't return only one slice for evaluation if there are 2 gpus, since it will try 
###          to send one slice to each gpu and then not send anything to gpu:1
###          otherwise error: check failed work_element_count > 0 (0 vs 0)
#    midslice = npimg[int(npimg.shape[0] / 2),:,:]
#    midseg   = npseg[int(npimg.shape[0] / 2),:,:]
#    return midslice[np.newaxis,:,:,np.newaxis], midseg[np.newaxis,:,:,np.newaxis]




def save_as_nii(img, svloc):
    if len(img.shape) == 4:
        to_sv = img[...,0]
    elif len(img.shape) == 3:
        to_sv = img
    else:
        print('failed to save : dimensions not clear')
        return
    to_sv_nii = nib.Nifti1Image(to_sv, None)
    to_sv_nii.to_filename(svloc)


###
###
### NOISE CONSTRUCTORS
###
### The following functions take data + parameters and
### and return a noisy image, with noise following the
### following options:
###     1. gaussian noise
###     2. uniform noise
###     3. sparse gaussian noise
###     4. sparse uniform noise
###     5. adversarial uniform noise
###     6. simulated physical noise (avg)
###     7. salt and pepper noise
###
###



##########
##########
### 1. ###
##########
##########

# noise follows N(0, eps/2)
# so that 95% of noise falls in [-eps,eps]
# then truncated back to [-1,1]
def make_gaussian_noise(data, eps=0.01):
    noise = np.random.normal(0, eps/2.0, data.shape)
    corrupted = data + noise
    return np.clip(corrupted, -1.0, 1.0)

##########
##########
### 2. ###
##########
##########

# noise follows Unif[-eps,eps]
def make_uniform_noise(data, eps=0.01):
    noise = np.random.uniform(-eps, eps, data.shape)
    corrupted = data + noise
    return np.clip(corrupted, -1.0, 1.0)

##########
##########
### 3. ###
##########
##########

# noise follows N(0, eps/2)
# then truncated as before,
# but on average only sparsity % of pixels are corrupted
def make_sparse_gaussian_noise(data, eps=0.01, sparsity=0.01):
    subset     = np.random.binomial(1, sparsity, size=data.shape)
    noise      = np.random.normal(0, eps/2.0, data.shape)
    noise      = np.multiply(subset, noise)
    corrupted  = data + noise
    return np.clip(corrupted, -1.0, 1.0)
    


##########
##########
### 4. ###
##########
##########

# noise follows Unif[-eps,eps]
# but only sparsity % of pixels are corrupted
def make_sparse_uniform_noise(data, eps=0.01, sparsity=0.01):
    subset     = np.random.binomial(1, sparsity, size=data.shape)
    noise      = np.random.uniform(-eps, eps, data.shape)
    noise      = np.multiply(subset, noise)
    corrupted  = data + noise
    return np.clip(corrupted, -1.0, 1.0)



##########
##########
### 7. ###
##########
##########

# noise is {-eps, eps}
# but only sparsity % of pixels are corrupted
def make_sparse_salt_noise(data, eps=0.01, sparsity=0.01):
    subset_p     = np.random.binomial(1, sparsity, size=data.shape)
    subset_m     = np.random.binomial(1, sparsity, size=data.shape)
    noise      = eps * (subset_p - subset_m)
    corrupted  = data + noise
    return np.clip(corrupted, -1.0, 1.0)



##########
##########
### 5. ###
##########
##########

# adversarial attack via FGSM
# (i.e. uniform-noise attack)
def fgsm(ximg, yseg, net, loss=dsc_l2, eps=options.delta):

#    x_adv   = np.zeros_like(ximg)
    perturb_adv = np.zeros_like(ximg)

    # need to batch, otherwise gradient gets too big to fit in memory
    nslices = ximg.shape[0]
    sss = [32*s for s in range(nslices//32)]
    if nslices%32 > 1:
        sss.append(nslices)

    for s in range(len(sss)-1):

        srt = sss[s]
        end = sss[s+1]
        x = ximg[srt:end,...]
        y = yseg[srt:end,...]

        yy = y.astype(np.float32)
        loss_vec = loss(yy, net.output)
        grad = K.gradients(loss_vec, net.input)
        gradfunc = K.function([net.input], grad)
        grad_at_x = gradfunc([x])
#        perturb_this = eps*np.sign(grad_at_x[0])
#        x_adv_this = x + perturb_this

#        x_adv[srt:end,...] = x_adv_this

        perturb_this = np.sign(grad_at_x[0])
        perturb_adv[srt:end,...] = perturb_this

#    return np.clip(x_adv, -1.0, 1.0)
    return perturb_adv


def make_adv_noise(data, seg=None, loaded_net=None, eps=options.delta):
    x_adv = fgsm(data, seg, loaded_net, eps=eps, loss=dsc_l2)
    return x_adv

def make_adv_noise_2(data, adv, eps):
    x_adv = data + eps*adv
    return np.clip(x_adv, -1.0, 1.0)


##########
##########
### 6. ###
##########
##########

# simulated physical noise
# takes a dataset of separate images not used in training (e.g. mda data)
# and for each HU value, finds the average of the (local) (empirical) noises
# over a ~10cm patch, then draws a sample from N(0, avg) as noise
def get_mda_imgs(dbfile  = '/rsrch1/ip/jacctor/livermask/trainingdata_small.csv', rootloc = '/rsrch1/ip/jacctor/LiTS/LiTS'):

    nscans  = 0
    nvalid  = 0
    nslices = 0

    with open(dbfile, 'r') as csvfile:
        myreader=csv.DictReader(csvfile, delimiter='\t')

        for row in myreader:
            imageloc = '%s/%s' % (rootloc, row['image'])
            truthloc = '%s/%s' % (rootloc, row['label'])
            print(imageloc, truthloc)
            nscans += 1
            try:
                npimg, header, npseg = preprocess.reorient(imageloc, segloc=truthloc)
                nslices += header['dim'][3]
                nvalid += 1
            except nib.filebasedimages.ImageFileError:
                print("could not read file")


    print('\ndone precomputing size:  ', nslices, ' slices, from ', nvalid, ' scans out of ', nscans, ' scans.\n')
    imgs = np.empty((nslices,npx,npx))
    segs = np.empty((nslices,npx,npx))

    sidx = 0

    with open(dbfile, 'r') as csvfile:
        myreader = csv.DictReader(csvfile, delimiter='\t')
        for row in myreader:
            imageloc = '%s/%s' % (rootloc, row['image'])
            truthloc = '%s/%s' % (rootloc, row['label'])

            print(imageloc, truthloc)

            try:
                npimg, header, npseg = preprocess.reorient(imageloc, segloc=truthloc)

                npimg = preprocess.resize_to_nn(npimg, transpose=True).astype(np.int16)
                npseg = preprocess.resize_to_nn(npseg, transpose=True).astype(np.uint8)

                sss = header['dim'][3]
                imgs[sidx:sidx+sss,...] = npimg
                segs[sidx:sidx+sss,...] = npseg
                sidx += sss
            except nib.filebasedimages.ImageFileError:
                print("ignoring the file I could't read earlier")

    return imgs, segs

# calculate local empirical noise on 2d slices
def get_noise_dist_2d(data, k=5):
    ker = np.ones((k,k))/(k**2) 
    mean = convolve2d(data,                   ker, mode='same', boundary='fill', fillvalue=0)
    var  = convolve2d(np.square(data - mean), ker, mode='same', boundary='fill', fillvalue=0)    
    return np.sqrt(var)

# calculate local empirical noise on 3d stack
# calculations are done 2d-slicewise to deal with anisotropic voxels
def get_noise_dist_3d(data3d, k=5):
    stdev = np.zeros_like(data3d)
    nslices = data3d.shape[0]
    for s in range(nslices):
        stdev[s] = get_noise_dist_2d(data3d[s,...], k=k)
    return stdev

# build lookup table for empirical noise
def compile_noise_dist(datax, datay, b=(100,10), r=[[-990,990],[0,60]]):
    print(datax.shape, datay.shape) #datax = pixel values, datay = stdevs
    t3 = time.time()
    h = np.histogram2d(datax.flatten(), datay.flatten(), bins=b, range=r)
    distlist = {hu_lb: 0.0, hu_ub: 0.0}
    for hu in range(1,h[0].shape[0]):
        hist_HU = h[0][hu,:]
        if sum(hist_HU):
            data_HU = [None]*int(sum(hist_HU))
            idx=0
            for i in range(h[0].shape[1]):
                hval = int(hist_HU[i])
                if hval > 0:
                    for jjj in range(hval):
                        data_HU[idx+jjj] = i+0.5
                idx += hval
            distlist[hu+hu_lb] = np.mean(data_HU) * 2.0/(hu_ub-hu_lb)
    t4 = time.time()
    print('time:', t4-t3)
    return distlist

###
### generate physical noise table : putting pieces together
###
def generate_physical_noise_distribution_table(dbfile_mda, rootloc_mda):
    imgs, _ = get_mda_imgs(dbfile=dbfile_mda, rootloc=rootloc_mda)
    imgs    = preprocess.window( imgs, hu_lb, hu_ub)
    stdev   = get_noise_dist_3d(imgs, k=5)
    dlist   = compile_noise_dist(imgs, stdev, b=(hu_ub-hu_lb,std_ub-std_lb), r=[[hu_lb, hu_ub],[std_lb, std_ub]])
    return dlist


# pixelwise noisemaker
def make_physical_noise_at_val(imgval, dlist):
    try:
        hu_val = int((imgval + 1.0)*(hu_ub-hu_lb)/2.0 + hu_lb)
        noisyval = np.random.normal(loc=imgval, scale=dlist[hu_val])
    except KeyError:
        noisyval = imgval
    return noisyval

# make a vectorized function
def make_physical_noise_vec(dlist):
    f = lambda x: make_physical_noise_at_val(x, dlist)
    return np.vectorize(f)

def make_physical_noise(img, dlist=None):
    noisy =  make_physical_noise_vec(dlist)(img)
    return np.clip(noisy, -1.0, 1.0)




###
### make functions, to loop over noise generation
###


def set_params(epslist=None, sparsity=0.001, loaded_net=None, seg=None, dlist=None, ns=5, adv=None):
    noisemaker_dict = { \
            'gaussian' : { \
                'name' : 'gaussian',
                'nsamples': ns,
                'maker': lambda x,e : make_gaussian_noise(x, eps=e), 
                'dsc_l2'  : [],
                'dsc_int' : [],
                'del_l2'  : [],
                'del_int' : [],
                'del_l2_reg'  : [],
                'del_int_reg' : [],
                'dsc_int_reg' : [],
                'dsc_l2_reg'  : [],
                'eps_list': [],
                'list' : epslist },
            'adversarial' : { \
                'name' : 'adversarial',
                'nsamples': 1,
                'maker': lambda x,e : make_adv_noise_2(x, adv, e),
                'del_l2'  : [],
                'del_int' : [],
                'dsc_int' : [],
                'dsc_l2'  : [],
                'del_l2_reg'  : [],
                'del_int_reg' : [],
                'dsc_int_reg' : [],
                'dsc_l2_reg'  : [],
                'eps_list': [],
                'list' : epslist },
            'physical' : { \
                'name' : 'physical',
                'nsamples': 1,
                'maker': lambda x,e : make_physical_noise(x, dlist=dlist), 
                'del_int' : [] ,
                'del_l2'  : [] ,
                'dsc_int' : [] ,
                'dsc_l2'  : [] ,
                'del_l2_reg'  : [],
                'del_int_reg' : [],
                'dsc_int_reg' : [],
                'dsc_l2_reg'  : [],
                'eps_list': [],
                'list' : [1]  },  
            }
    return noisemaker_dict

def make_all_noisy_images(imgloc, segloc, epslist, sparsity, loaded_net, loaded_net_reg, dlist, rootsaveloc='./', ns=5):
    
    img, seg = get_img(imgloc, segloc)

    os.makedirs(rootsaveloc, exist_ok=True)
    os.makedirs(rootsaveloc+'/clean', exist_ok=True)
    save_as_nii(img, rootsaveloc+'/clean/img.nii.gz')
    save_as_nii(seg, rootsaveloc+'/clean/seg.nii.gz')

    print('\n\n')
    print(imgloc)
    print(segloc)
    print('\t making clean prediction...')
    clean_pred     = loaded_net.predict(img, batch_size=16)
    clean_pred_seg = (clean_pred >= 0.5).astype(np.uint8)
    clean_pred_seg = preprocess.largest_connected_component(clean_pred_seg).astype(np.uint8)
    clean_dsc_int  = dsc_int_3D(clean_pred_seg, seg)
    clean_dsc_l2   = dsc_l2_3D(clean_pred, seg)
    save_as_nii(clean_pred,     rootsaveloc+'/clean/pred.nii.gz')
    save_as_nii(clean_pred_seg, rootsaveloc+'/clean/pred-seg.nii.gz')

    print('\t making clean regularized prediction...')
    clean_pred_reg     = loaded_net_reg.predict(img, batch_size=16)
    clean_pred_seg_reg = (clean_pred_reg >= 0.5).astype(np.uint8)
    clean_pred_seg_reg = preprocess.largest_connected_component(clean_pred_seg_reg).astype(np.uint8)
    clean_dsc_int_reg  = dsc_int_3D(clean_pred_seg_reg, seg)
    clean_dsc_l2_reg   = dsc_l2_3D(clean_pred_reg, seg)
    save_as_nii(clean_pred_reg,     rootsaveloc+'/clean/pred-reg.nii.gz')
    save_as_nii(clean_pred_seg_reg, rootsaveloc+'/clean/pred-seg-reg.nii.gz')

    print('\t generating noise structures...')
    adv = fgsm(img, seg, loaded_net, eps=1.0, loss=dsc_l2)
    noisemaker = set_params(epslist=epslist, seg=seg, sparsity=sparsity, loaded_net=loaded_net, dlist=dlist, ns=ns, adv=adv)

    for distr in noisemaker:
      
        f      = noisemaker[distr]['maker']
        e_vals = noisemaker[distr]['list']

        for ie, e in enumerate(e_vals):
            t1 = time.time()
            print(noisemaker[distr]['name'], e)
            if e == 0.0:
                # update values in noisemaker dict
                noisemaker[distr]['eps_list'].append(e)
                noisemaker[distr]['dsc_l2' ].append(clean_dsc_l2)
                noisemaker[distr]['dsc_int'].append(clean_dsc_int)
                noisemaker[distr]['del_l2' ].append(0.0)
                noisemaker[distr]['del_int'].append(0.0)
                noisemaker[distr]['dsc_l2_reg' ].append(clean_dsc_l2_reg)
                noisemaker[distr]['dsc_int_reg'].append(clean_dsc_int_reg)
                noisemaker[distr]['del_l2_reg' ].append(0.0)
                noisemaker[distr]['del_int_reg'].append(0.0)

#                # save image files as nii
#                print('\t saving nii files...')
#                saveloc = rootsaveloc+'/'+noisemaker[distr]['name']+'/'+str(e)+'/'
#                os.makedirs(saveloc, exist_ok=True)
#                noise = np.zeros_like(img)
#                save_as_nii(noise,              saveloc+'perturbation.nii.gz')
#                save_as_nii(img,                saveloc+'noisy-img.nii.gz')
#                save_as_nii(clean_pred,         saveloc+'noisy-pred-float.nii.gz')
#                save_as_nii(clean_pred_reg,     saveloc+'noisy-pred-float-reg.nii.gz')
#                save_as_nii(clean_pred_seg,     saveloc+'noisy-pred-seg.nii.gz')
#                save_as_nii(clean_pred_seg_reg, saveloc+'noisy-pred-seg-reg.nii.gz')
            else:
                for sss in range(noisemaker[distr]['nsamples']):
                
                    # make noise
                    print('\t', e, sss)
                    print('\t making noise...')
                    noisy = f(img, e)

                    # predict on the noisy images, no regularization
                    print('\t making noisy prediction...')
                    noisy_pred     = loaded_net.predict(noisy, batch_size=16)
                    noisy_pred_seg = (noisy_pred >= 0.5).astype(np.uint8)
                    noisy_pred_seg = preprocess.largest_connected_component(noisy_pred_seg).astype(np.uint8)

                    # predict on the noisy images, with regularization
                    print('\t making regularized prediction...')
                    noisy_pred_reg     = loaded_net_reg.predict(noisy, batch_size=16)
                    noisy_pred_seg_reg = (noisy_pred_reg >= 0.5).astype(np.uint8)
                    noisy_pred_seg_reg = preprocess.largest_connected_component(noisy_pred_seg_reg).astype(np.uint8)

                    # calculate and print DSC scores, no regularization
                    print('\t calculating dsc scores...')
                    noisy_dsc_int = dsc_int_3D(noisy_pred_seg, seg)
                    noisy_dsc_l2  = dsc_l2_3D(noisy_pred, seg)
                    print('\t\t clean     DSC(int) : ', clean_dsc_int, 
                            '\t noisy     DSC(int) : ', noisy_dsc_int, 
                            '\t delta : ', noisy_dsc_int - clean_dsc_int)
                    print('\t\t clean     DSC(l2)  : ', clean_dsc_l2,  
                            '\t noisy     DSC(l2)  : ', noisy_dsc_l2,  
                            '\t delta : ', noisy_dsc_l2  - clean_dsc_l2)

                    # calculate and print DSC scores, with regularization
                    print('\t calculating regularized dsc scores...')
                    noisy_dsc_int_reg = dsc_int_3D(noisy_pred_seg_reg, seg)
                    noisy_dsc_l2_reg  = dsc_l2_3D(noisy_pred_reg, seg)
                    print('\t\t clean reg DSC(int) : ', clean_dsc_int_reg, 
                            '\t noisy reg DSC(int) : ', noisy_dsc_int_reg, 
                            '\t delta : ', noisy_dsc_int_reg - clean_dsc_int_reg)
                    print('\t\t clean reg DSC(l2)  : ', clean_dsc_l2_reg,  
                            '\t noisy reg DSC(l2)  : ', noisy_dsc_l2_reg, 
                            '\t delta : ', noisy_dsc_l2_reg  - clean_dsc_l2_reg)

                    # update values in noisemaker dict
                    print('\t storing data into dict...')
                    noisemaker[distr]['eps_list'].append(e)
                    noisemaker[distr]['dsc_l2' ].append(noisy_dsc_l2)
                    noisemaker[distr]['dsc_int'].append(noisy_dsc_int)
                    noisemaker[distr]['del_l2' ].append(noisy_dsc_l2  - clean_dsc_l2)
                    noisemaker[distr]['del_int'].append(noisy_dsc_int - clean_dsc_int)
                    noisemaker[distr]['dsc_l2_reg' ].append(noisy_dsc_l2_reg)
                    noisemaker[distr]['dsc_int_reg'].append(noisy_dsc_int_reg)
                    noisemaker[distr]['del_l2_reg' ].append(noisy_dsc_l2_reg  - clean_dsc_l2_reg)
                    noisemaker[distr]['del_int_reg'].append(noisy_dsc_int_reg - clean_dsc_int_reg)

                    # save image files as nii
                    saveloc = rootsaveloc+'/'+noisemaker[distr]['name']+'/'+str(e)+'/'
                    os.makedirs(saveloc, exist_ok=True)
                    print('\t saving nii files...')
                    save_as_nii(noisy - img,        saveloc+'perturbation-'         +str(sss)+'.nii.gz')
                    save_as_nii(noisy,              saveloc+'noisy-img-'            +str(sss)+'.nii.gz')
                    save_as_nii(noisy_pred,         saveloc+'noisy-pred-float-'     +str(sss)+'.nii.gz')
                    save_as_nii(noisy_pred_reg,     saveloc+'noisy-pred-float-reg-' +str(sss)+'.nii.gz')
                    save_as_nii(noisy_pred_seg,     saveloc+'noisy-pred-seg-'       +str(sss)+'.nii.gz')
                    save_as_nii(noisy_pred_seg_reg, saveloc+'noisy-pred-seg-reg-'   +str(sss)+'.nii.gz')

                    print('\t deleting files...')
                    del noisy
                    del noisy_pred
                    del noisy_pred_reg
                    del noisy_pred_seg
                    del noisy_pred_seg_reg

            t2 = time.time()
            print('\t time : ', t2-t1)

    print('\n')
    return noisemaker


net_train = load_model(options.model,     custom_objects={'dsc_l2':dsc_l2, 'dsc':dsc, 'dsc_int':dsc_int, 'ISTA':ISTA})
net_reg   = load_model(options.model_reg, custom_objects={'dsc_l2':dsc_l2, 'dsc':dsc, 'dsc_int':dsc_int, 'ISTA':ISTA})

try:
    print(' loading dlist...')
    savedlist = open(options.outdir+'dlist.json', 'r')
    dlist = json.load(savedlist)
    savedlist.close()
except:
    print(' could not find dlist - now generating...')
    dlist =  generate_physical_noise_distribution_table(dbfile_mda, rootloc_mda)
    print(' saving dlist to .json file...')
    savedlist = open(options.outdir+'dlist.json', 'w+')
    json.dump(dlist, savedlist)
    savedlist.close()


if options.test:
    locdict =    { \
        124: { \
            'vol': '/rsrch1/ip/jacctor/LiTS/LiTS/TrainingBatch2/volume-124.nii',
            'seg': '/rsrch1/ip/jacctor/LiTS/LiTS/TrainingBatch2/segmentation-124.nii' },
        120: {  \
           'vol': '/rsrch1/ip/jacctor/LiTS/LiTS/TrainingBatch2/volume-120.nii',
           'seg': '/rsrch1/ip/jacctor/LiTS/LiTS/TrainingBatch2/segmentation-120.nii' }, }
elif options.one_at_a_time:
    locdict = {}
    base_vol_string = '/rsrch1/ip/jacctor/LiTS/LiTS/TrainingBatch2/volume-'
    base_seg_string = '/rsrch1/ip/jacctor/LiTS/LiTS/TrainingBatch2/segmentation-'
    i = options.idxOne
    locdict[i] = { 'vol' : base_vol_string+str(i)+'.nii', 'seg': base_seg_string+str(i)+'.nii' }
else:
    locdict = {}
    base_vol_string = '/rsrch1/ip/jacctor/LiTS/LiTS/TrainingBatch2/volume-'
    base_seg_string = '/rsrch1/ip/jacctor/LiTS/LiTS/TrainingBatch2/segmentation-'
    for i in range(111, 132):
        locdict[i] = { 'vol' : base_vol_string+str(i)+'.nii', 'seg': base_seg_string+str(i)+'.nii' }

combined_dict = { \
    'gaussian' : { \
        'dsc_l2'  : [],
        'dsc_int' : [],
        'del_l2'  : [],
        'del_int' : [],
        'del_l2_reg'  : [],
        'del_int_reg' : [],
        'dsc_int_reg' : [],
        'dsc_l2_reg'  : [],
        'eps_list': [], },
    'adversarial' : { \
        'del_l2'  : [],
        'del_int' : [],
        'dsc_int' : [],
        'dsc_l2'  : [],
        'del_l2_reg'  : [],
        'del_int_reg' : [],
        'dsc_int_reg' : [],
        'dsc_l2_reg'  : [],
        'eps_list': [], },
    'physical' : { \
        'del_int' : [] ,
        'del_l2'  : [] ,
        'dsc_int' : [] ,
        'dsc_l2'  : [] ,
        'del_l2_reg'  : [],
        'del_int_reg' : [],
        'dsc_int_reg' : [],
        'dsc_l2_reg'  : [],
        'eps_list': [], },
    }

if options.test:
    outdirloc = options.outdir + 'test-noisemaker'
    epslist = [0.0, 0.01]
    nsamples=1
else:
    outdirloc = options.outdir + 'noisemaker'
    epslist = [0.000, 0.010, 0.020, 0.035, 0.050, 0.0750, 0.100, 0.125, 0.150, 0.175, 0.200]
    epslist = [round(j, 3) for j in epslist]
    nsamples=3
#    epslist = [0.0, 0.01]
#    nsamples=1

for im_idx in locdict:

    imgloc = locdict[im_idx]['vol']
    segloc = locdict[im_idx]['seg']
    nm = make_all_noisy_images(imgloc, segloc, epslist, 0.1, net_train, net_reg, dlist, rootsaveloc=outdirloc+'/'+str(im_idx), ns=nsamples)
    print('\n\n')
    
    for di in combined_dict:
        for listname in combined_dict[di]:
            combined_dict[di][listname] += nm[di][listname]

print(' saving results to json...')
savedict = open(outdirloc+'/'+str(options.idxOne)+'/noisemaker_results.json', 'w+')
json.dump(combined_dict, savedict)
savedict.close()




def plot_changes(combined_dict):
    print(' plotting...')
    iii = 1
    nrows = len(combined_dict) -1
    for distr in combined_dict:
        if distr is not "physical":
            plt.subplot(nrows,2,iii)
            plt.scatter(combined_dict[distr]['eps_list'], combined_dict[distr]['dsc_l2'],      c='b', alpha=0.5, s=9, marker='o')
            plt.scatter(combined_dict[distr]['eps_list'], combined_dict[distr]['dsc_l2_reg'],  c='r', alpha=0.5, s=9, marker='o')
            iii += 1
            plt.subplot(nrows,2,iii)
            plt.scatter(combined_dict[distr]['eps_list'], combined_dict[distr]['del_l2'],      c='b', alpha=0.5, s=9, marker='o')
            plt.scatter(combined_dict[distr]['eps_list'], combined_dict[distr]['del_l2_reg'],  c='r', alpha=0.5, s=9, marker='o')
            iii += 1
    plt.show()

    iii = 1
    for distr in combined_dict:
        if  distr is not "physical":    
            plt.subplot(nrows,2,iii)
            plt.scatter(combined_dict[distr]['eps_list'], combined_dict[distr]['dsc_int'],     c='b', alpha=0.5, s=9, marker='o')
            plt.scatter(combined_dict[distr]['eps_list'], combined_dict[distr]['dsc_int_reg'], c='r', alpha=0.5, s=9, marker='o')
            iii += 1
            plt.subplot(nrows,2,iii)
            plt.scatter(combined_dict[distr]['eps_list'], combined_dict[distr]['del_int'],     c='b', alpha=0.5, s=9, marker='o')
            plt.scatter(combined_dict[distr]['eps_list'], combined_dict[distr]['del_int_reg'], c='r', alpha=0.5, s=9, marker='o')
            iii += 1
    plt.show()

#plot_changes(combined_dict)



