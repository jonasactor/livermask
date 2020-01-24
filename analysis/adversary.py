import numpy as np
import os
from optparse import OptionParser
import matplotlib as mptlib
mptlib.use('TkAgg')
import matplotlib.pyplot as plt

import keras
import keras.backend as K
from keras import models
from keras import layers
from keras.models import load_model
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

import tensorflow as tf

from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import FastGradientMethod

import sys
sys.setrecursionlimit(5000)
sys.path.append('liverhcc')
from mymetrics import dsc_l2, dsc, dsc_int, l1
from ista import ISTA
import preprocess

###
### set options
###
parser = OptionParser()
parser.add_option( "--model",
        action="store", dest="model", default=None,
        help="model location", metavar="PATH")
parser.add_option( "--outdir",
        action="store", dest="outdir", default="./",
        help="out location", metavar="PATH")
parser.add_option( "--imgloc",
        action="store", dest="imgloc", default='/rsrch1/ip/jacctor/LiTS/LiTS/TrainingBatch2/volume-50.nii',
        help="img location", metavar="PATH.nii")
parser.add_option( "--segloc",
        action="store", dest="segloc", default='/rsrch1/ip/jacctor/LiTS/LiTS/TrainingBatch2/segmentation-50.nii',
        help="seg location", metavar="PATH.nii")
(options, args) = parser.parse_args()



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
    npimg           = preprocess.resize_to_nn(npimg, transpose=True).astype(np.int16)
    npimg           = preprocess.window(npimg, -100,300)
    npimg           = preprocess.rescale(npimg, -100,300)
    npseg           = preprocess.resize_to_nn(npseg, transpose=True).astype(np.uint8)

    print(npimg.shape)
    print(npseg.shape)
    assert npimg.shape == npseg.shape
    midslice = npimg[int(npimg.shape[0] / 2),:,:]
    midseg   = npseg[int(npimg.shape[0] / 2),:,:]

    return midslice[np.newaxis,:,:,np.newaxis], midseg[np.newaxis,:,:,np.newaxis]

x_test, y_test = get_img(options.imgloc, options.segloc)

#mu = 0.00
mu = 0.001  # proxy for some form of spectral regularization, even though there's no conv here

IMG_DTYPE = np.int16
SEG_DTYPE = np.uint8

_globalexpectedpixel=512
_nx = 256
_ny = 256


###
### load model and perform adversarial attack
###

net = load_model(options.model, custom_objects={'dsc_l2':dsc_l2, 'l1':l1, 'dsc':dsc, 'dsc_int':dsc, 'ISTA':ISTA})


wrapper = KerasModelWrapper(net)

x = tf.placeholder(tf.float32, shape=(None, _ny, _nx, 1)) 
y = tf.placeholder(tf.float32, shape=(None, _ny, _nx, 1)) 

fgsm = FastGradientMethod(wrapper, sess=session)

fgsm_eps = 0.1
fgsm_min = 0.0
fgsm_max = 1.0
fgsm_parameters = {'eps':fgsm_eps, 'clip_min':fgsm_min, 'clip_max':fgsm_max }
adversary = fgsm.generate(x, **fgsm_parameters)
adversary = tf.stop_gradient(adversary)
adv_prob = net(adversary) # struct for output of adv feedforward

fetches = [adv_prob]
fetches.append(adversary)
outputs = session.run(fetches=fetches, feed_dict={x:x_test})
adv_prob = outputs[0]
adv_examples = outputs[1]
#adv_predicted = adv_prob.argmax(1)
#adv_accuracy = np.mean(adv_predicted==y_test)
#print("accuracy:\t %.5f" % adv_accuracy)

#n_classes = 10
#f, ax = plt.subplots(2,5,figsize=(10,5))
#ax = ax.flatten()
#for i in range(n_classes):
#    diff = adv_examples[i] - test[i]
#    norm_diff = np.linalg.norm(diff)
#    max_diff  = np.abs(diff).max()
#    print('norm: ', norm_diff, ' max abs val: ', max_diff)
#    ax[i].imshow(diff.reshape(28,28))
#plt.show()
#
#f,ax = plt.subplots(2,5,figsize=(10,5))
#ax = ax.flatten()
#for i in range(n_classes):
#    ax[i].imshow(adv_examples[i].reshape(28,28))
#    ax[i].set_title("adv: %d, label: %d" % (adv_predicted[i], y_test[i]))
#plt.show()
