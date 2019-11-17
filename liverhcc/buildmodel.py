import numpy as np
import keras
from keras.layers import Input, Conv2D, UpSampling2D, Lambda, SpatialDropout2D, Dense, Layer, Activation, BatchNormalization, MaxPool2D, concatenate, LocallyConnected2D, AveragePooling2D
from keras.models import Model, Sequential
from keras.models import model_from_json, load_model
from keras.utils import multi_gpu_model
from keras.utils.np_utils import to_categorical
import keras.backend as K
from keras.initializers import Constant
import tensorflow as tf

import settings


def addConvBNSequential(model, filters=32):
    if settings.options.batchnorm:
          model = BatchNormalization()(model)
    if settings.options.dropout > 0.0:
          model = SpatialDropout2D(settings.options.dropout)(model)
    if settings.options.rescon:
        model = concatenate([model, Conv2D(filters=filters,kernel_size=(3,3), padding='same', activation=settings.options.activation)(model)])
    else:
        model = Conv2D(filters=filters, kernel_size=(3,3), padding='same', activation=settings.options.activation)(model)
    return model

def module_down(model, filters=16, activation='prelu'):
    for i in range(settings.options.nu):
        model = addConvBNSequential(model, filters=filters)
    model = AveragePooling2D()(model)
#    model = MaxPool2D()(model)
    return model

def module_up(model, filters=16):
    if settings.options.reverse_up:
        for i in range(settings.options.nu):
            model = addConvBNSequential(model, filters=filters)
        model = UpSampling2D()(model)
    else:
        model = UpSampling2D()(model)
        for i in range(settings.options.nu):
            model = addConvBNSequential(model, filters=filters)
    return model


def module_mid(model, depth, filters=16):
    if settings.options.fanout and depth < settings.options.depth:
        filters = filters*2
    if depth==0:
        for i in range(settings.options.nu_bottom):
            model = addConvBNSequential(model, filters=filters)
        return model
    else:
        m_down = module_down(model, filters=filters)
        m_mid  = module_mid(m_down, depth=depth-1, filters=filters)
        m_up   = module_up(m_mid, filters=filters)
        if settings.options.skip:
            m_up = concatenate([model, m_up])
        else:
            m_up = m_down + m_up
        return m_up

def get_unet( _num_classes=1):
    _depth   = settings.options.depth
    _filters = settings.options.filters
    _v       = settings.options.v
    _nu      = settings.options.nu

    layer_in  = Input(shape=(settings._ny,settings._nx,1))
    layer_mid = Activation('linear')(layer_in)

    for j in range(_v):
        layer_mid = module_mid(layer_mid, depth=_depth, filters=_filters)
        for i in range(_nu):
            layer_mid = addConvBNSequential(layer_mid,      filters=_filters)

    layer_out = Dense(_num_classes, activation='sigmoid', use_bias=True)(layer_mid)
    model = Model(inputs=layer_in, outputs=layer_out)
    if settings.options.gpu > 1:
        return multi_gpu_model(model, gpus=settings.options.gpu)
    else:
        return model






