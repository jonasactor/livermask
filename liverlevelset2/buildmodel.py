import numpy as np
import keras
from keras.layers import Input, Conv2D, UpSampling2D, Lambda, SpatialDropout2D, Dense, Layer, Activation, BatchNormalization, MaxPool2D, concatenate, LocallyConnected2D, AveragePooling2D, DepthwiseConv2D, Add, Maximum, Minimum, Multiply
from keras.models import Model, Sequential
from keras.models import model_from_json, load_model
from keras.utils import multi_gpu_model
from keras.utils.np_utils import to_categorical
from keras.regularizers import l1, l2
import keras.backend as K
from keras.initializers import Constant
import tensorflow as tf

import settings
from ista import ISTA


def make_kernel(a):
    a = np.asarray(a)
    a = a.reshape(list(a.shape) + [1,1])
    return K.constant(a)

class ForcingFunction(Layer):

    def __init__(self, in_img, **kwargs):
        self.img = in_img
        self.dt = 0.05
        self.mXN = [[0,0,0],[-1,1,0],[0,0,0]]
        self.mXP = [[0,0,0],[0,-1,1],[0,0,0]]
        self.mYN = [[0,0,0],[0,1,0],[0,-1,0]]
        self.mYP = [[0,1,0],[0,-1,0],[0,0,0]]
        self.mXC = [[0,0,0],[-1,0,1],[0,0,0]]
        self.mYC = [[0,1,0],[0,0,0],[0,-1,0]]
        self.mLX = [[1,0,-1],[2,0,-2],[1,0,-1]]
        self.mLY = [[1,2,1],[0,0,0],[-1,-2,-1]]
        self.kXP = make_kernel(mXP) # first-order FD in x direction with prior information
        self.kXN = make_kernel(mXN) # first-order FD in x direction with ahead information
        self.kYP = make_kernel(mYP) # first-order FD in y direction with prior information
        self.kYN = make_kernel(mYN) # first-order FD in y direction with ahead information
        self.kXC = make_kernel(mXC) # second-order centered FD in x direction
        self.kYC = make_kernel(mYC) # second-order centered FD in y direction
        self.kLX = make_kernel(mLX) # Canny dx kernel
        self.kLY = make_kernel(mLY) # Canny dy kernel
        self.o   = make_kernel([[1]])
        super(ForcingFunction, self).__init__(**kwargs)

    def build(self, input_shape):
        self.alpha = self.add_weight(name='alpha',   # curvature coeff
            shape=(1,1,1,1),
            initializer='ones',
            trainable=False)
        self.beta  = self.add_weight(name='beta',    # transport coeff
            shape=(1,1,1,1),
            initializer='ones',
            trainable=False)
        self.gamma = self.add_weight(name='gamma',   # balloon coeff
            shape=(1,1,1,1),
            initializer='ones',
            trainable=False)
        if not settings.options.true_lse:
            self.kernel_edges = self.add_weight(name='edge_kernel',
                shape=(3,3,1,16),
                initializer='normal',
                trainable=True)
            self.kernel_kappa_1 = self.add_weight(name='kappa_kernel_1',
                shape=(3,3,1,16),
                initializer='normal',
                trainable=True)
            self.kernel_kappa_2 = self.add_weight(name='kappa_kernel_2',
                shape=(3,3,16,48),
                initializer='normal',
                trainable=True)

        ### edge detection
        ### g_I(x) = 1 / (1 + norm(grad(I))^2 ), slightly modified

        if settings.options.true_lse:
            self.edges_x = K.conv2d(self.img, self.kLX, padding='same')
            self.edges_y = K.conv2d(self.img, self.kLY, padding='same')
            self.edges = K.square(self.edges_x) + K.square(self.edges_y)
            self.edges = K.square(self.edges)
            self.edges = edges/K.max(self.edges)
            self.edges = K.constant(1.0) / (self.edges + K.constant(1.0))
            ### grad( edge_detection )
            self.grad_edges_x = K.conv2d(self.edges, self.kXC, padding='same')
            self.grad_edges_y = K.conv2d(self.edges, self.kYC, padding='same')


        super(ForcingFunction, self).build(input_shape)

    def call(self, u):
        ### edge detection
        ### g_I(x) = 1 / (1 + norm(grad(I))^2 ), slightly modified
        if not settings.options.true_lse:
            edges = K.conv2d(self.img, self.kernel_edges, padding='same')
            edges = K.square(edges)
            edges = K.sum(edges, axis=-1, keepdims=True)
            edges = K.square(edges)
            edges = edges/K.max(edges)
            edges = K.constant(1.0) / (edges + K.constant(1.0))
            ### grad( edge_detection )
            grad_edges_x = K.conv2d(edges, self.kXC, padding='same')
            grad_edges_y = K.conv2d(edges, self.kYC, padding='same')
        else:
            edges = self.edges
            grad_edges_x = self.grad_edges_x
            grad_edges_y = self.grad_edges_y


        ### transport - upwind approx to grad( edge_detection)^T grad( u )
        xp = K.conv2d(u, self.kXP, padding='same')
        xn = K.conv2d(u, self.kXN, padding='same')
        yp = K.conv2d(u, self.kYP, padding='same')
        yn = K.conv2d(u, self.kYN, padding='same')
        fxp =         K.relu(       grad_edges_x)
        fxn =  -1.0 * K.relu(-1.0 * grad_edges_x)
        fyp =         K.relu(       grad_edges_y)
        fyn =  -1.0 * K.relu(-1.0 * grad_edges_y)
        xpp = fxp*xp
        xnn = fxn*xn
        ypp = fyp*yp
        ynn = fyn*yn
        xterms = xpp + xnn
        yterms = ypp + ynn
        transport = 20.0*(xterms + yterms)

        ### curvature kappa( u )
        if settings.options.true_lse:
            gradu_x = K.conv2d(u, self.kXC, padding='same')
            gradu_y = K.conv2d(u, self.kYC, padding='same')
            normu = K.square(gradu_x) + K.square(gradu_y)
            normu = K.sqrt(normu + K.epsilon())
            kappa_x = gradu_x / (normu + K.epsilon())
            kappa_x = K.conv2d(kappa_x, self.kXC, padding='same')
            kappa_y = gradu_y / (normu + K.epsilon())
            kappa_y = K.conv2d(kappa_y, self.kYC, padding='same')
            kappa = kappa_x + kappa_y
        else:
            gradu = K.conv2d(u, self.kernel_kappa_1, padding='same')
            normu = K.square(gradu)
            normu = K.sum(normu, axis=-1, keepdims=True)
            normu = K.sqrt(normu + K.epsilon())
            gradu = gradu / (normu + K.epsilon())
            gradu = K.conv2d(gradu, self.kernel_kappa_2, padding='same')
            kappa = K.sum(gradu)

        curvature = edges*kappa*normu

        ### balloon force
        balloon = edges*normu


        return u + K.constant(self.dt)*(\
               curvature * self.alpha \
            +  transport * self.beta  \
            +  balloon   * self.gamma )

    def compute_output_shape(self, input_shape):
        return input_shape



def get_unet(_nt=settings.options.nt, _final_sigma='sigmoid'):

    in_imgs   = Input(shape=(settings.options.trainingresample,settings.options.trainingresample,1))

    img = Activation('linear')(in_img) # img
    mid_layer = Lambda(lambda x : (x > 2.0*(90.0-settings.options.hu_lb)/(settings.options.hu_ub - settings.options.hu_lb) -1)\
                                * (x < 2.0*(110.0-settings.options.hu_lb)/(settings.options.hu_ub - settings.options.hu_lb) -1))(in_imgs) # u0

    # Forcing Function F depends on image and on  u, but not on time
    F = ForcingFunction(img)
    for ttt in range(_nt):
        mid_layer = F(mid_layer)

    out_layer = Activation('linear')(mid_layer)
    model = Model(in_imgs, out_layer)
    if settings.options.gpu > 1:
        return multi_gpu_model(model, gpus=settings.options.gpu)
    else:
        return model
