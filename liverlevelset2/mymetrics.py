import numpy as np
import keras
import keras.backend as K
import tensorflow as tf



###
###
### Similarity scores and metrics
###

def dsc(y_true, y_pred, smooth=0.0000001):
    num = 2.0*K.sum(K.abs(y_true*y_pred), axis=(1,2)) + smooth
    den = K.sum(K.abs(y_true), axis=(1,2)) + K.sum(K.abs(y_pred), axis=(1,2)) + smooth
    return 1.0 - (num/den)

def dsc_l2(y_true, y_pred, smooth=0.0000001):
    num = K.sum(K.square(y_true - y_pred), axis=(1,2)) + smooth
    den = K.sum(K.square(y_true), axis=(1,2)) + K.sum(K.square(y_pred), axis=(1,2)) + smooth
    return num/den

def dsc_int(y_true, y_pred, smooth=0.0000001):
    y_pred_int = 0.5*(K.sign(y_pred - 0.5)+1)
    y_true_int = 0.5*(K.sign(y_true - 0.5)+1)
    num = 2.0*K.sum(K.abs(y_true_int*y_pred_int), axis=(1,2))
    den = K.sum(K.abs(y_true_int), axis=(1,2)) + K.sum(K.abs(y_pred_int), axis=(1,2)) + smooth
    return 1.0 - (num/den)

def dsc_int_3D(y_true, y_pred, smooth=0.0000001):
    y_pred_int = 0.5*(np.sign(y_pred - 0.5)+1)
    y_true_int = 0.5*(np.sign(y_true - 0.5)+1)
    num = 2.0*np.sum(np.abs(y_true_int*y_pred_int))
    den = np.sum(np.abs(y_true_int)) + np.sum(np.abs(y_pred_int)) + smooth
    return num/den

def dsc_l2_3D(y_true, y_pred, smooth=0.0000001):
    num = np.sum(np.square(y_true - y_pred)) 
    den = np.sum(np.square(y_true)) + np.sum(np.square(y_pred)) + smooth
    return num/den

def l1(y_true, y_pred, smooth=0.00001):
    return K.sum(K.abs(y_true-y_pred), axis=(1,2))/(256*256)

def dsc_crossentropy(y_true, y_pred, smooth=0.00000001):
    K.set_epsilon(1e-8)
    num = K.binary_crossentropy(y_true, y_pred)
    den = ( K.sum( K.abs(y_true), axis=(1,2,3)) + smooth ) * K.ones_like(y_true)
    return 0.5 * 256.0*256.0 * num/den

def dsc_l2_convex(y_true, y_pred, smooth=0.00000001):
    num = K.sum(K.square(y_true - y_pred), axis=(1,2))
    den = K.sum( K.square(y_true), axis=(1,2)) + smooth
    return 0.5 * num / den
