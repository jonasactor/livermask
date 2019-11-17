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

def make_kernel(a):
    a = np.asarray(a)
    a = a.reshape(list(a.shape) + [1,1])
    return K.constant(a)

# matrices
mXN = [[0,0,0],[-1,1,0],[0,0,0]]
mXP = [[0,0,0],[0,-1,1],[0,0,0]]
mYN = [[0,0,0],[0,1,0],[0,-1,0]]
mYP = [[0,1,0],[0,-1,0],[0,0,0]]
mXC = [[0,0,0],[-0.5,0,0.5],[0,0,0]]
mYC = [[0,-0.5,0],[0,0,0],[0,0.5,0]]
mLX = [[1,0,-1],[2,0,-2],[1,0,-1]]
mLY = [[1,2,1],[0,0,0],[-1,-2,-1]]

# kernels
kXP = make_kernel(mXP)
kXN = make_kernel(mXN)
kYP = make_kernel(mYP)
kYN = make_kernel(mYN)
kXC = make_kernel(mXC)
kYC = make_kernel(mYC)
kLX = make_kernel(mLX)
kLY = make_kernel(mLY)
o   = make_kernel([[1]])

def scale(x, shape, reshape, alpha):
        x = K.reshape(x, reshape)
        x = K.batch_dot(x, alpha, axes=[2,1])
        x = K.reshape(x, shape)
        return x

class LevelSetStep(Layer):

    def __init__(self, in_img, in_dims, **kwargs):
        self.img = in_img
        self.dt = 0.05
        self.dims = in_dims
        self.dx = in_dims[:,0,0,:]
        self.dy = in_dims[:,1,0,:]
        self.dz = in_dims[:,2,0,:]
        self.shp = K.shape(in_img)
        self.rhp = (self.shp[0], self.shp[1]*self.shp[2], self.shp[3])
        super(LevelSetStep, self).__init__(**kwargs)

    def build(self, input_shape):
        self.alpha = self.add_weight( name        = 'alpha'   ,
                                      shape       = (1,1,1,1) ,
                                      initializer = 'ones'    ,
                                      trainable   = True      )
        self.beta  = self.add_weight( name        = 'beta'    ,
                                      shape       = (1,1,1,1) ,
                                      initializer = 'ones'    ,
                                      trainable   = True      )
        self.gamma = self.add_weight( name        = 'gamma'   ,
                                      shape       = (1,1,1,1) ,
                                      initializer = 'ones'    ,
                                      trainable   = True      )
        super(LevelSetStep, self).build(input_shape)

    def call(self, ins):
        u   = ins[0]
        img = ins[1]

        ### grad(I)
        i_x = K.conv2d(self.img, kLX, padding='same')
#        i_x = scale(i_x, self.shp, self.rhp, self.dx)
        i_y = K.conv2d(self.img, kLY, padding='same')
#        i_y = scale(i_y, self.shp, self.rhp, self.dy)
        norm_grad_i = K.square(i_x) + K.square(i_y)

        ### g = 1. / ( 1. + 20*norm(grad(I))^4
        g = K.square(norm_grad_i)
        g = g/K.max(g)
        g = K.constant(1.0) / (20.0*g + K.constant(1.0))

        ### grad(g)
        g_x = K.conv2d(g, kXC, padding='same')
#        g_x = scale(g_x, self.shp, self.rhp, self.dx)
        g_y = K.conv2d(g, kYC, padding='same')
#        g_y = scale(g_y, self.shp, self.rhp, self.dy)

        ### transport
        xp = K.conv2d(u, kXP, padding='same')
        xn = K.conv2d(u, kXN, padding='same')
        yp = K.conv2d(u, kYP, padding='same')
        yn = K.conv2d(u, kYN, padding='same')
        fxp =        K.relu(        g_x)
        fxn = -1.0 * K.relu( -1.0 * g_x)
        fyp =        K.relu(        g_y)
        fyn = -1.0 * K.relu( -1.0 * g_y)
        xpp = fxp*xp
        xnn = fxn*xn
        ypp = fyp*yp
        ynn = fyn*yn
        xterms = xpp + xnn
#        xterms = scale(xterms, self.shp, self.rhp, self.dx)
        yterms = ypp + ynn
#        yterms = scale(yterms, self.shp, self.rhp, self.dy)
        transport = xterms + yterms

        ### curvature
        u_x = K.conv2d(u, kXC, padding='same')
#        u_x = scale(u_x, self.shp, self.rhp, self.dx)
        u_y = K.conv2d(u, kYC, padding='same')
#        u_y = scale(u_y, self.shp, self.rhp, self.dy)
        norm_grad_u = K.sqrt( K.epsilon() + K.square(u_x) + K.square(u_y) )
        u_x = u_x / (norm_grad_u + K.epsilon())
        u_y = u_y / (norm_grad_u + K.epsilon())
        kappa_x = K.conv2d(u_x, kXC, padding='same')
#        kappa_x = scale(kappa_x, self.shp, self.rhp, self.dx)
        kappa_y = K.conv2d(u_y, kYC, padding='same')
#        kappa_y = scale(kappa_y, self.shp, self.rhp, self.dy)
        kappa = kappa_x + kappa_y
        curvature = g*kappa*norm_grad_u

        ### balloon
        balloon = g*norm_grad_u

        return u + K.constant(self.dt)*(   curvature * self.alpha \
                                         + transport * self.beta  \
                                         + balloon   * self.gamma )

    def compute_output_shape(self, input_shape):
        return input_shape




class LearnedEdge(Layer):

    def __init__(self, in_img, in_dims, **kwargs):
        self.img = in_img
        self.dt = 0.05
        super(LearnedEdge, self).__init__(**kwargs)

    def build(self, input_shape):
        self.alpha = self.add_weight( name        = 'alpha'   ,
                                      shape       = (1,1,1,1) ,
                                      initializer = 'ones'    ,
                                      trainable   = True      )
        self.beta  = self.add_weight( name        = 'beta'    ,
                                      shape       = (1,1,1,1) ,
                                      initializer = 'ones'    ,
                                      trainable   = True      )
        self.gamma = self.add_weight( name        = 'gamma'   ,
                                      shape       = (1,1,1,1) ,
                                      initializer = 'ones'    ,
                                      trainable   = True      )
        self.kernel_1   = self.add_weight( name        = 'kernel_1'    ,
                                           shape       = (3,3,1,16)    ,
                                           initializer = 'normal'      ,
                                           trainable   = True          )
        self.bias_1     = self.add_weight( name        = 'bias_1'      ,
                                           shape       = (16,)         ,
                                           initializer = 'zeros'       ,
                                           trainable   = True          )
        self.kernel_2   = self.add_weight( name        = 'kernel_2'    ,
                                           shape       = (3,3,16,16)   ,
                                           initializer = 'normal'      ,
                                           trainable   = True          )
        self.bias_2     = self.add_weight( name        = 'bias_2'      ,
                                           shape       = (16,)         ,
                                           initializer = 'zeros'       ,
                                           trainable   = True          )
        self.kernel_3   = self.add_weight( name        = 'kernel_3'    ,
                                           shape       = (3,3,16,16)   ,
                                           initializer = 'normal'      ,
                                           trainable   = True          )
        self.bias_3     = self.add_weight( name        = 'bias_3'      ,
                                           shape       = (16,)         ,
                                           initializer = 'zeros'       ,
                                           trainable   = True          )
        self.kernel_4   = self.add_weight( name        = 'kernel_4'    ,
                                           shape       = (3,3,16,16)   ,
                                           initializer = 'normal'      ,
                                           trainable   = True          )
        self.bias_4     = self.add_weight( name        = 'bias_4'      ,
                                           shape       = (16,)         ,
                                           initializer = 'zeros'       ,
                                           trainable   = True          )
        self.kernel_5   = self.add_weight( name        = 'kernel_5'    ,
                                           shape       = (3,3,16,32)   ,
                                           initializer = 'normal'      ,
                                           trainable   = True          )
        self.bias_5     = self.add_weight( name        = 'bias_5'      ,
                                           shape       = (32,)         ,
                                           initializer = 'zeros'       ,
                                           trainable   = True          )
        self.kernel_6   = self.add_weight( name        = 'kernel_6'    ,
                                           shape       = (3,3,32,32)   ,
                                           initializer = 'normal'      ,
                                           trainable   = True          )
        self.bias_6     = self.add_weight( name        = 'bias_6'      ,
                                           shape       = (32,)         ,
                                           initializer = 'zeros'       ,
                                           trainable   = True          )
        self.kernel_7   = self.add_weight( name        = 'kernel_7'    ,
                                           shape       = (3,3,32,32)   ,
                                           initializer = 'normal'      ,
                                           trainable   = True          )
        self.bias_7     = self.add_weight( name        = 'bias_7'      ,
                                           shape       = (32,)         ,
                                           initializer = 'zeros'       ,
                                           trainable   = True          )
        self.kernel_8   = self.add_weight( name        = 'kernel_8'    ,
                                           shape       = (3,3,32,16)   ,
                                           initializer = 'normal'      ,
                                           trainable   = True          )
        self.bias_8     = self.add_weight( name        = 'bias_8'      ,
                                           shape       = (16,)         ,
                                           initializer = 'zeros'       ,
                                           trainable   = True          )
        self.kernel_9   = self.add_weight( name        = 'kernel_9'    ,
                                           shape       = (3,3,32,16)   ,
                                           initializer = 'normal'      ,
                                           trainable   = True          )
        self.bias_9     = self.add_weight( name        = 'bias_9'      ,
                                           shape       = (16,)         ,
                                           initializer = 'zeros'       ,
                                           trainable   = True          )
        self.kernel_10  = self.add_weight( name        = 'kernel_10'   ,
                                           shape       = (3,3,16,16)   ,
                                           initializer = 'normal'      ,
                                           trainable   = True          )
        self.bias_10    = self.add_weight( name        = 'bias_10'     ,
                                           shape       = (16,)         ,
                                           initializer = 'zeros'       ,
                                           trainable   = True          )
        self.kernel_11  = self.add_weight( name        = 'kernel_11'   ,
                                           shape       = (3,3,32,16)   ,
                                           initializer = 'normal'      ,
                                           trainable   = True          )
        self.bias_11     = self.add_weight( name       = 'bias_11'     ,
                                           shape       = (16,)         ,
                                           initializer = 'zeros'       ,
                                           trainable   = True          )
        self.kernel_12  = self.add_weight( name        = 'kernel_12'   ,
                                           shape       = (3,3,16,1)    ,
                                           initializer = 'normal'      ,
                                           trainable   = True          )
        self.bias_12     = self.add_weight( name       = 'bias_12'     ,
                                           shape       = (1,)          ,
                                           initializer = 'zeros'       ,
                                           trainable   = True          )
        super(LearnedEdge, self).build(input_shape)

    def call(self, ins):
        u   = ins[0]
        img = ins[1]

        ### g learned by net
        g1 = K.conv2d(self.img, self.kernel_1, padding='same')
        g1 = K.bias_add(g1, self.bias_1)
        g1 = K.relu(g1)
        g1 = K.conv2d(g1, self.kernel_2, padding='same')
        g1 = K.bias_add(g1, self.bias_2)
        g1 = K.relu(g1)
        g2 = K.pool2d(g1, (2,2), padding='same')
        g2 = K.conv2d(g2, self.kernel_3, padding='same')
        g2 = K.bias_add(g2, self.bias_3)
        g2 = K.relu(g2)
        g2 = K.conv2d(g2, self.kernel_4, padding='same')
        g2 = K.bias_add(g2, self.bias_4)
        g2 = K.relu(g2)
        g3 = K.pool2d(g2, (2,2), padding='same')
        g3 = K.conv2d(g3, self.kernel_5, padding='same')
        g3 = K.bias_add(g3, self.bias_5)
        g3 = K.relu(g3)
        g3 = K.conv2d(g3, self.kernel_6, padding='same')
        g3 = K.bias_add(g3, self.bias_6)
        g3 = K.relu(g3)
        g3 = K.conv2d(g3, self.kernel_7, padding='same')
        g3 = K.bias_add(g3, self.bias_7)
        g3 = K.relu(g3)
        g3 = K.conv2d(g3, self.kernel_8, padding='same')
        g3 = K.bias_add(g3, self.bias_8)
        g3 = K.relu(g3)
        g4 = K.resize_images(g3, 2, 2, data_format='channels_last', interpolation='bilinear')
        g4 = K.concatenate([g2, g3])
        g4 = K.conv2d(g4, self.kernel_9, padding='same')
        g4 = K.bias_add(g4, self.bias_9)
        g4 = K.relu(g4)
        g4 = K.conv2d(g4, self.kernel_10, padding='same')
        g4 = K.bias_add(g4, self.bias_10)
        g5 = K.relu(g4)
        g5 = K.resize_images(g5, 2, 2, data_format='channels_last', interpolation='bilinear')
        g5 = K.concatenate([g1, g4])
        g5 = K.conv2d(g5, self.kernel_11, padding='same')
        g5 = K.bias_add(g5, self.bias_11)
        g5 = K.relu(g5)
        g5 = K.conv2d(g5, self.kernel_12, padding='same')
        g5 = K.bias_add(g5, self.bias_12)
        g  = K.sigmoid(g5)


        ### grad(g)
        g_x = K.conv2d(g, kXC, padding='same')
#        g_x = scale(g_x, self.shp, self.rhp, self.dx)
        g_y = K.conv2d(g, kYC, padding='same')
#        g_y = scale(g_y, self.shp, self.rhp, self.dy)

        ### transport
        xp = K.conv2d(u, kXP, padding='same')
        xn = K.conv2d(u, kXN, padding='same')
        yp = K.conv2d(u, kYP, padding='same')
        yn = K.conv2d(u, kYN, padding='same')
        fxp =        K.relu(        g_x)
        fxn = -1.0 * K.relu( -1.0 * g_x)
        fyp =        K.relu(        g_y)
        fyn = -1.0 * K.relu( -1.0 * g_y)
        xpp = fxp*xp
        xnn = fxn*xn
        ypp = fyp*yp
        ynn = fyn*yn
        xterms = xpp + xnn
#        xterms = scale(xterms, self.shp, self.rhp, self.dx)
        yterms = ypp + ynn
#        yterms = scale(yterms, self.shp, self.rhp, self.dy)
        transport = xterms + yterms

        ### curvature
        u_x = K.conv2d(u, kXC, padding='same')
#        u_x = scale(u_x, self.shp, self.rhp, self.dx)
        u_y = K.conv2d(u, kYC, padding='same')
#        u_y = scale(u_y, self.shp, self.rhp, self.dy)
        norm_grad_u = K.sqrt( K.epsilon() + K.square(u_x) + K.square(u_y) )
        u_x = u_x / (norm_grad_u + K.epsilon())
        u_y = u_y / (norm_grad_u + K.epsilon())
        kappa_x = K.conv2d(u_x, kXC, padding='same')
#        kappa_x = scale(kappa_x, self.shp, self.rhp, self.dx)
        kappa_y = K.conv2d(u_y, kYC, padding='same')
#        kappa_y = scale(kappa_y, self.shp, self.rhp, self.dx)
        kappa = kappa_x + kappa_y
        curvature = g*kappa*norm_grad_u

        ### balloon
        balloon = g*norm_grad_u

        return u + K.constant(self.dt)*(   curvature * self.alpha \
                                         + transport * self.beta  \
                                         + balloon   * self.gamma )

    def compute_output_shape(self, input_shape):
        return input_shape



class LSNStep(Layer):

    def __init__(self, in_img, in_dims, **kwargs):
        self.img = in_img
        self.dt = 0.05
        super(LSNStep, self).__init__(**kwargs)

    def build(self, input_shape):
        self.alpha = self.add_weight( name        = 'alpha'   ,
                                      shape       = (1,1,1,1) ,
                                      initializer = 'ones'    ,
                                      trainable   = True      )
        self.beta  = self.add_weight( name        = 'beta'    ,
                                      shape       = (1,1,1,1) ,
                                      initializer = 'ones'    ,
                                      trainable   = True      )
        self.gamma = self.add_weight( name        = 'gamma'   ,
                                      shape       = (1,1,1,1) ,
                                      initializer = 'ones'    ,
                                      trainable   = True      )
        self.kernel_1   = self.add_weight( name        = 'kernel_1'    ,
                                           shape       = (3,3,1,16)    ,
                                           initializer = 'normal'      ,
                                           trainable   = True          )
        self.bias_1     = self.add_weight( name        = 'bias_1'      ,
                                           shape       = (16,)         ,
                                           initializer = 'zeros'       ,
                                           trainable   = True          )
        self.kernel_2   = self.add_weight( name        = 'kernel_2'    ,
                                           shape       = (3,3,16,16)   ,
                                           initializer = 'normal'      ,
                                           trainable   = True          )
        self.bias_2     = self.add_weight( name        = 'bias_2'      ,
                                           shape       = (16,)         ,
                                           initializer = 'zeros'       ,
                                           trainable   = True          )
        self.kernel_3   = self.add_weight( name        = 'kernel_3'    ,
                                           shape       = (3,3,16,16)   ,
                                           initializer = 'normal'      ,
                                           trainable   = True          )
        self.bias_3     = self.add_weight( name        = 'bias_3'      ,
                                           shape       = (16,)         ,
                                           initializer = 'zeros'       ,
                                           trainable   = True          )
        self.kernel_4   = self.add_weight( name        = 'kernel_4'    ,
                                           shape       = (3,3,16,16)   ,
                                           initializer = 'normal'      ,
                                           trainable   = True          )
        self.bias_4     = self.add_weight( name        = 'bias_4'      ,
                                           shape       = (16,)         ,
                                           initializer = 'zeros'       ,
                                           trainable   = True          )
        self.kernel_5   = self.add_weight( name        = 'kernel_5'    ,
                                           shape       = (3,3,16,32)   ,
                                           initializer = 'normal'      ,
                                           trainable   = True          )
        self.bias_5     = self.add_weight( name        = 'bias_5'      ,
                                           shape       = (32,)         ,
                                           initializer = 'zeros'       ,
                                           trainable   = True          )
        self.kernel_6   = self.add_weight( name        = 'kernel_6'    ,
                                           shape       = (3,3,32,32)   ,
                                           initializer = 'normal'      ,
                                           trainable   = True          )
        self.bias_6     = self.add_weight( name        = 'bias_6'      ,
                                           shape       = (32,)         ,
                                           initializer = 'zeros'       ,
                                           trainable   = True          )
        self.kernel_7   = self.add_weight( name        = 'kernel_7'    ,
                                           shape       = (3,3,32,32)   ,
                                           initializer = 'normal'      ,
                                           trainable   = True          )
        self.bias_7     = self.add_weight( name        = 'bias_7'      ,
                                           shape       = (32,)         ,
                                           initializer = 'zeros'       ,
                                           trainable   = True          )
        self.kernel_8   = self.add_weight( name        = 'kernel_8'    ,
                                           shape       = (3,3,32,16)   ,
                                           initializer = 'normal'      ,
                                           trainable   = True          )
        self.bias_8     = self.add_weight( name        = 'bias_8'      ,
                                           shape       = (16,)         ,
                                           initializer = 'zeros'       ,
                                           trainable   = True          )
        self.kernel_9   = self.add_weight( name        = 'kernel_9'    ,
                                           shape       = (3,3,32,16)   ,
                                           initializer = 'normal'      ,
                                           trainable   = True          )
        self.bias_9     = self.add_weight( name        = 'bias_9'      ,
                                           shape       = (16,)         ,
                                           initializer = 'zeros'       ,
                                           trainable   = True          )
        self.kernel_10  = self.add_weight( name        = 'kernel_10'   ,
                                           shape       = (3,3,16,16)   ,
                                           initializer = 'normal'      ,
                                           trainable   = True          )
        self.bias_10    = self.add_weight( name        = 'bias_10'     ,
                                           shape       = (16,)         ,
                                           initializer = 'zeros'       ,
                                           trainable   = True          )
        self.kernel_11  = self.add_weight( name        = 'kernel_11'   ,
                                           shape       = (3,3,32,16)   ,
                                           initializer = 'normal'      ,
                                           trainable   = True          )
        self.bias_11     = self.add_weight( name       = 'bias_11'     ,
                                           shape       = (16,)         ,
                                           initializer = 'zeros'       ,
                                           trainable   = True          )
        self.kernel_12  = self.add_weight( name        = 'kernel_12'   ,
                                           shape       = (3,3,16,1)    ,
                                           initializer = 'normal'      ,
                                           trainable   = True          )
        self.bias_12     = self.add_weight( name       = 'bias_12'     ,
                                           shape       = (1,)          ,
                                           initializer = 'zeros'       ,
                                           trainable   = True          )
        self.kernel_k1  = self.add_weight( name        = 'kernel_k1'   ,
                                           shape       = (3,3,1,16)    ,
                                           initializer = 'normal'      ,
                                           trainable   = True          )
        self.kernel_k2  = self.add_weight( name        = 'kernel_k2'   ,
                                           shape       = (3,3,16,1)    ,
                                           initializer = 'normal'      ,
                                           trainable   = True          )
        super(LSNStep, self).build(input_shape)

    def call(self, ins):
        u   = ins[0]
        img = ins[1]

        ### g learned by net
        g1 = K.conv2d(self.img, self.kernel_1, padding='same')
        g1 = K.bias_add(g1, self.bias_1)
        g1 = K.relu(g1)
        g1 = K.conv2d(g1, self.kernel_2, padding='same')
        g1 = K.bias_add(g1, self.bias_2)
        g1 = K.relu(g1)
        g2 = K.pool2d(g1, (2,2), padding='same')
        g2 = K.conv2d(g2, self.kernel_3, padding='same')
        g2 = K.bias_add(g2, self.bias_3)
        g2 = K.relu(g2)
        g2 = K.conv2d(g2, self.kernel_4, padding='same')
        g2 = K.bias_add(g2, self.bias_4)
        g2 = K.relu(g2)
        g3 = K.pool2d(g2, (2,2), padding='same')
        g3 = K.conv2d(g3, self.kernel_5, padding='same')
        g3 = K.bias_add(g3, self.bias_5)
        g3 = K.relu(g3)
        g3 = K.conv2d(g3, self.kernel_6, padding='same')
        g3 = K.bias_add(g3, self.bias_6)
        g3 = K.relu(g3)
        g3 = K.conv2d(g3, self.kernel_7, padding='same')
        g3 = K.bias_add(g3, self.bias_7)
        g3 = K.relu(g3)
        g3 = K.conv2d(g3, self.kernel_8, padding='same')
        g3 = K.bias_add(g3, self.bias_8)
        g3 = K.relu(g3)
        g4 = K.resize_images(g3, 2, 2, data_format='channels_last', interpolation='bilinear')
        g4 = K.concatenate([g2, g4])
        g4 = K.conv2d(g4, self.kernel_9, padding='same')
        g4 = K.bias_add(g4, self.bias_9)
        g4 = K.relu(g4)
        g4 = K.conv2d(g4, self.kernel_10, padding='same')
        g4 = K.bias_add(g4, self.bias_10)
        g4 = K.relu(g4)
        g5 = K.resize_images(g4, 2, 2, data_format='channels_last', interpolation='bilinear')
        g5 = K.concatenate([g1, g5])
        g5 = K.conv2d(g5, self.kernel_11, padding='same')
        g5 = K.bias_add(g5, self.bias_11)
        g5 = K.relu(g5)
        g5 = K.conv2d(g5, self.kernel_12, padding='same')
        g5 = K.bias_add(g5, self.bias_12)
        g  = K.sigmoid(g5)

        ### grad(g)
        g_x = K.conv2d(g, kXC, padding='same')
        g_x = scale(g_x, self.shp, self.rhp, self.dx)
        g_y = K.conv2d(g, kYC, padding='same')
        g_y = scale(g_y, self.shp, self.rhp, self.dy)

        ### transport - upwind
        xp = K.conv2d(u, xKP, padding='same')
        xn = K.conv2d(u, xKN, padding='same')
        yp = K.conv2d(u, xYP, padding='same')
        yn = K.conv2d(u, xYN, padding='same')
        fxp =        K.relu(        g_x)
        fxn = -1.0 * K.relu( -1.0 * g_x)
        fyp =        K.relu(        g_y)
        fyn = -1.0 * K.relu( -1.0 * g_y)
        xpp = fxp*xp
        xnn = fxn*xn
        ypp = fyp*yp
        ynn = fyn*yn
        xterms = xpp + xnn
        xterms = scale(xterms, self.shp, self.rhp, self.dx)
        yterms = ypp + ynn
        yterms = scale(yterms, self.shp, self.rhp, self.dy)
        transport = xterms + yterms

        ### curvature - learned
        grad_u = K.conv2d(u, self.kernel_k1, padding='same')
        norm_grad_u = K.sqrt(   K.epsilon() + K.sum( K.square(grad_u), axis=-1, keepdims=True) )
        grad_u = grad_u / (norm_grad_u + K.epsilon())
        kappa = K.conv2d(grad_u, self.kernel_k2, padding='same')
        curvature = g*kappa*norm_grad_u

        ### balloon
        balloon = g*norm_grad_u

        return u + K.constant(self.dt)*(   curvature * self.alpha \
                                         + transport * self.beta  \
                                         + balloon   * self.gamma )

    def compute_output_shape(self, input_shape):
        return input_shape










def get_lse(_nt, _final_sigma='sigmoid'):

    in_imgs = Input(shape=(settings._ny,settings._nx,2))
    in_dims = Input(shape=(3,1,1))

    #### I
    in_img = Lambda(lambda x : x[...,0][...,None])(in_imgs)
    img = Activation('linear')(in_img)

    ### u0
    in_layer = Lambda(lambda x: x[...,1][...,None])(in_imgs)
    mid_layer = Activation('linear')(in_layer)

    F = LevelSetStep(img, in_dims)
    for ttt in range(_nt):
        mid_layer = F([mid_layer, img])

    out_layer = Activation(_final_sigma)(mid_layer)
    model = Model([in_imgs, in_dims], out_layer)

    if settings.options.gpu > 1:
        return multi_gpu_model(model, gpus=settings.options.gpu)
    else:
        return model



def get_lse_G(_nt, _final_sigma='sigmoid'):

    in_imgs = Input(shape=(settings._ny,settings._nx,2))
    in_dims = Input(shape=(3,1,1))

    #### I
    in_img = Lambda(lambda x : x[...,0][...,None])(in_imgs)
    img = Activation('linear')(in_img)

    ### u0
    in_layer = Lambda(lambda x: x[...,1][...,None])(in_imgs)
    mid_layer = Activation('linear')(in_layer)

    F = LearnedEdge(img, in_dims)
    for ttt in range(_nt):
        mid_layer = F([mid_layer, img])

    out_layer = Activation(_final_sigma)(mid_layer)
    model = Model([in_imgs, in_dims], out_layer)

    if settings.options.gpu > 1:
        return multi_gpu_model(model, gpus=settings.options.gpu)
    else:
        return model

def get_lsn(_nt, _final_sigma='sigmoid'):

    in_imgs = Input(shape=(settings._ny,settings._nx,2))
    in_dims = Input(shape=(3,1,1))

    #### I
    in_img = Lambda(lambda x : x[...,0][...,None])(in_imgs)
    img = Activation('linear')(in_img)

    ### u0
    in_layer = Lambda(lambda x: x[...,1][...,None])(in_imgs)
    mid_layer = Activation('linear')(in_layer)

    F = LSNStep(img, in_dims)
    for ttt in range(_nt):
        mid_layer = F([mid_layer, img])

    out_layer = Activation(_final_sigma)(mid_layer)
    model = Model([in_imgs, in_dims], out_layer)

    if settings.options.gpu > 1:
        return multi_gpu_model(model, gpus=settings.options.gpu)
    else:
        return model








