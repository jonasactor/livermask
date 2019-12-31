import numpy as np
import keras
from keras.layers import Input, Conv2D, UpSampling2D, Lambda, SpatialDropout2D, Dense, Layer, Activation, BatchNormalization, MaxPool2D, concatenate, LocallyConnected2D, AveragePooling2D, DepthwiseConv2D, Add
from keras.models import Model, Sequential
from keras.models import model_from_json, load_model
from keras.utils import multi_gpu_model
from keras.utils.np_utils import to_categorical
from keras.regularizers import l1
import keras.backend as K
from keras.initializers import Constant
import tensorflow as tf

import settings

def DepthwiseConvBlock(model_in):
    if settings.options.fanout:
        _dm = 2
    else:
        _dm = 1
    if settings.options.batchnorm:
          model_in = BatchNormalization()(model_in)
#    if settings.options.dropout > 0.0:
#          model_in = SpatialDropout2D(settings.options.dropout)(model_in)
    if settings.options.regularizer:
        model = DepthwiseConv2D( \
            kernel_size=(3,3),
            padding='same', 
            depth_multiplier=_dm, 
            activation=settings.options.activation, 
            depthwise_regularizer=l1(settings.options.regularizer)  )(model_in)
    else:
        model = DepthwiseConv2D( \
            kernel_size=(3,3),
            padding='same', 
            depth_multiplier=_dm, 
            activation=settings.options.activation  )(model_in)
    if settings.options.rescon and settings.options.fanout: 
        model = concatenate([model_in, model])
    elif settings.options.rescon and not settings.options.fanout:
        model = Add()([model_in, model])
    return model
    
def ConvBlock(model_in, filters=32):
    if settings.options.batchnorm:
          model_in = BatchNormalization()(model_in)
#    if settings.options.dropout > 0.0:
#          model_in = SpatialDropout2D(settings.options.dropout)(model_in)
    if settings.options.regularizer:
        model = Conv2D( \
            filters=filters, 
            kernel_size=(3,3), 
            padding='same', 
            activation=settings.options.activation,
            kernel_regularizer=l1(settings.options.regularizer) )(model_in)
    else:
        model = Conv2D( \
            filters=filters, 
            kernel_size=(3,3), 
            padding='same', 
            activation=settings.options.activation )(model_in)
    if settings.options.rescon and settings.options.fanout: 
        model = concatenate([model, model_in])
    elif settings.options.rescon and not settings.options.fanout:
        model = Add()([model_in, model])
    return model

def addConvBNSequential(model, filters=32):
    if settings.options.batchnorm:
          model = BatchNormalization()(model)
#    if settings.options.dropout > 0.0:
#          model = SpatialDropout2D(settings.options.dropout)(model)
    if settings.options.rescon: 
        model = concatenate([model, Conv2D(filters=filters,kernel_size=(3,3), padding='same', activation=settings.options.activation)(model)])
    if settings.options.rescon2: 
        model_conv = Conv2D(filters=filters,kernel_size=(3,3), padding='same', activation=settings.options.activation)(model)
        model = concatenate([model, model_conv])
    else:
        model = Conv2D(filters=filters, kernel_size=(3,3), padding='same', activation=settings.options.activation)(model)
    return model

def module_down(model, filters=16, activation='prelu'):
    for i in range(settings.options.nu):
        model = DepthwiseConvBlock(model)
    model = AveragePooling2D()(model)
#    model = MaxPool2D()(model)
    return model

def module_up(model, filters=16):
    if settings.options.reverse_up:
        for i in range(settings.options.nu):
            model = DepthwiseConvBlock(model)
        model = UpSampling2D()(model)
    else:
        model = UpSampling2D()(model)
        for i in range(settings.options.nu):
            model = DepthwiseConvBlock(model)
    return model


def module_mid(model, depth, filters=16):
    if settings.options.fanout and depth < settings.options.depth:
        filters = filters*2
    if depth==0:
        for i in range(settings.options.nu_bottom):
#            model = addConvBNSequential(model, filters=filters)
            model = DepthwiseConvBlock(model)
        return model
    else:
        m_down = module_down(model, filters=filters)
        m_mid  = module_mid(m_down, depth=depth-1, filters=filters)
        m_up   = module_up(m_mid, filters=filters)
        if settings.options.skip:
            m_up = concatenate([model, m_up])
        else:
            m_up = Add()([model, m_up])
        return m_up

def get_unet( _num_classes=1):
    _depth   = settings.options.depth
    _filters = settings.options.filters
    _v       = settings.options.v
    _nu      = settings.options.nu

    layer_in  = Input(shape=(settings._ny,settings._nx,1))
    layer_mid = Activation('linear')(layer_in)

    for j in range(_v):
        for i in range(_nu):
            layer_mid = ConvBlock(layer_mid,            filters=_filters)        
        layer_mid = module_mid(layer_mid, depth=_depth, filters=_filters)
        for i in range(_nu):
            layer_mid = DepthwiseConvBlock(layer_mid)

    if settings.options.dropout > 0.0:
        layer_mid = SpatialDropout2D(settings.options.dropout)(layer_mid)
    layer_out = Dense(_num_classes, activation='sigmoid', use_bias=True)(layer_mid)
    model = Model(inputs=layer_in, outputs=layer_out)
    if settings.options.gpu > 1:
        return multi_gpu_model(model, gpus=settings.options.gpu)
    else:
        return model






